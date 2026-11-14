"""Maximum-detail traces must be joined to the actual update, not guessed from a batch."""
import base64
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import archx
from archx.container.reader import Reader
from archx.container.schema import BlockType
from archx.container.values import Timeline
from archx.adapters.example_gradients import read_example_gradients


def test_dense_history_examples_and_sgd_update(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / 'deep.archx.mdl'
    recorder = archx.attach(model, opt, str(path), full_history=True, max_rel_error=0,
                            meta={'run_config': {'seed': 42, 'dataset': 'test'}})
    states = [{name: p.detach().clone().numpy() for name, p in model.named_parameters()}]
    records = []
    for step in range(3):
        x = torch.tensor([[1., float(step)], [2., -1.]])
        y = torch.tensor([[1.], [0.]])
        opt.zero_grad()
        parts = ((model(x) - y) ** 2).reshape(-1) / 2
        parts.sum().backward(retain_graph=True)
        r = recorder.trace_examples(parts, [b'example-a,target=1', b'example-b,target=0'])
        assert r['examples'] == 2
        sample_grads = []
        for part in parts:
            direct = torch.autograd.grad(part, list(model.parameters()), retain_graph=True)
            sample_grads.append({name: grad.detach().cpu().numpy().copy()
                                 for (name, _), grad in zip(model.named_parameters(), direct)})
        records.append(({name: p.grad.detach().clone().numpy() for name, p in model.named_parameters()},
                        sample_grads))
        recorder.log(loss=parts.sum().item(), data=x)
        opt.step()
        states.append({name: p.detach().clone().numpy() for name, p in model.named_parameters()})
    recorder.close()
    reader = Reader(path)
    timeline = Timeline(reader)
    assert timeline.steps == [0, 1, 2, 3]
    assert len(list(reader.blocks(BlockType.ANCHOR))) == 1
    assert len(list(reader.blocks(BlockType.DELTA))) == 3
    assert reader.meta()['torch_initial_seed'] == 42
    assert reader.meta()['run_config']['dataset'] == 'test'
    assert reader.meta()['optimizer_groups'][0]['lr'] == 0.1
    for i in range(4):
        for name, expected in states[i].items():
            np.testing.assert_allclose(timeline.reconstruct(i)[name], expected, atol=1e-7, rtol=0)
    traces = list(read_example_gradients(reader))
    assert [s for s, _, _ in traces] == [1, 2, 3]
    for step, meta, gradients in traces:
        assert base64.b64decode(meta['examples_base64'][0]) == b'example-a,target=1'
        assert meta['max_aggregate_abs_error'] < 1e-6
        for name, sample in gradients.items():
            np.testing.assert_allclose(sample.sum(axis=0), records[step-1][0][name], atol=1e-6)
            for i in range(2):
                np.testing.assert_allclose(sample[i], records[step-1][1][i][name], atol=1e-7)
            np.testing.assert_allclose(states[step][name] - states[step-1][name],
                                       -0.1 * sample.sum(axis=0), atol=1e-7)


def test_trace_rejects_incorrect_loss_and_budget(tmp_path):
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    r = archx.attach(model, opt, str(tmp_path / 'fail.archx.mdl'), full_history=True)
    parts = model(torch.tensor([[1.], [2.]])).reshape(-1)
    parts.sum().backward(retain_graph=True)
    with pytest.raises(ValueError, match='budget'):
        r.trace_examples(parts, [b'a', b'b'], max_bytes=1)
    with pytest.raises(ValueError, match='do not sum'):
        r.trace_examples(parts * .5, [b'a', b'b'])
    with pytest.raises(ValueError, match='one bytes example'):
        r.trace_examples(parts, [b'a'])
    r.close()


def test_trace_requires_dense_mode(tmp_path):
    m = torch.nn.Linear(1, 1); opt = torch.optim.SGD(m.parameters(), lr=.1)
    r = archx.attach(m, opt, str(tmp_path / 'adaptive.archx.mdl'), full_history=False)
    with pytest.raises(ValueError, match='full_history'):
        r.trace_examples(torch.ones(1), [b'a'])
    r.close()


def test_shuffle_lossless_and_legacy_and_new_blocks(tmp_path):
    import json
    import io
    from archx.adapters.example_gradients import (shuffle_trace_payload, unshuffle_trace_payload,
                                                  legacy_trace_payload)
    from archx.container.writer import Writer
    # Include negative zero, subnormals, repeated zeros and arbitrary finite bit patterns.
    bits = np.array([0, 0x80000000, 1, 0x3f800000, 0xbf800000, 0x7f7fffff], dtype=np.uint32)
    arrays = {'dense': bits.view(np.float32).reshape(2, 3),
              'zeros': np.zeros((2, 512), dtype=np.float32)}
    arrays['zeros'].view(np.uint32).flat[3] = 0x80000000  # negative zero is not discarded
    arrays['zeros'].view(np.uint32).flat[-1] = 1  # smallest subnormal
    examples = [b'\x00\xff\x01', ('Data changes weights. مثال متن 🧠\n' * 800).encode('utf-8')]
    meta = {'version': 1, 'meaning': 'local gradients', 'names': list(arrays),
            'examples_base64': [base64.b64encode(x).decode() for x in examples],
            'loss_parts': [0., 1.], 'max_aggregate_abs_error': 0.,
            'raw_gradient_bytes': sum(v.nbytes for v in arrays.values())}
    legacy = legacy_trace_payload(meta, arrays)
    shuffled = shuffle_trace_payload(legacy)
    rebuilt_meta, rebuilt = unshuffle_trace_payload(shuffled)
    assert rebuilt_meta['tensors'][1]['kind'] == 'sparse'
    assert [base64.b64decode(s) for s in rebuilt_meta['examples_base64']] == examples
    for name in arrays:
        assert rebuilt[name].dtype == arrays[name].dtype
        assert rebuilt[name].shape == arrays[name].shape
        assert rebuilt[name].tobytes() == arrays[name].tobytes()
    path = tmp_path / 'both.archx.mdl'
    with Writer(path) as writer:
        writer.write_block(BlockType.EXAMPLE_GRADIENTS, legacy, step=1)
        writer.write_block(BlockType.EXAMPLE_GRADIENTS_SHUFFLED, shuffled, step=2)
    rows = list(read_example_gradients(Reader(path)))
    assert [r[0] for r in rows] == [1, 2]
    assert len(list(read_example_gradients(Reader(path), step=2))) == 1
    for _, got_meta, got in rows:
        assert got_meta['examples_base64'] == meta['examples_base64']
        for name in arrays:
            assert got[name].tobytes() == arrays[name].tobytes()
    # A crash/recovery must retain a complete new-style block without requiring
    # earlier trace blocks or an external compression dictionary.
    from archx.container.recovery import recover_in_place
    torn = tmp_path / 'torn.archx.mdl'
    writer = Writer(torn)
    writer.write_block(BlockType.EXAMPLE_GRADIENTS_SHUFFLED, shuffled, step=2)
    writer.abort()
    recover_in_place(torn)
    recovered = list(read_example_gradients(Reader(torn)))
    assert len(recovered) == 1
    for name in arrays:
        assert recovered[0][2][name].tobytes() == arrays[name].tobytes()


def test_shuffled_trace_corruption_is_rejected():
    import io
    import json
    from archx.adapters.example_gradients import unshuffle_trace_payload
    meta = {'version': 2, 'encoding': 'byte_shuffle_4', 'names': [], 'tensors': [],
            'example_lengths': [3]}
    b = io.BytesIO()
    np.savez(b, metadata=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
             examples=np.frombuffer(b'x', dtype=np.uint8))
    with pytest.raises(ValueError, match='invalid example lengths'):
        unshuffle_trace_payload(b.getvalue())
    meta = {'version': 2, 'encoding': 'adaptive_float32_v1', 'names': ['x'],
            'tensors': [{'shape': [10**9], 'dtype': '<f4', 'kind': 'sparse', 'nnz': 0}],
            'example_lengths': []}
    b = io.BytesIO()
    np.savez(b, metadata=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
             examples=np.array([], dtype=np.uint8), g0=np.array([], dtype=np.uint8))
    with pytest.raises(ValueError, match='decoded size limit'):
        unshuffle_trace_payload(b.getvalue(), max_output_bytes=1024)


def test_deep_trace_waits_when_writer_falls_behind_without_losing_steps(tmp_path, monkeypatch):
    import threading
    from archx.recorder import Recorder
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = Recorder._handle

    def slow_first_trace(self, writer, item):
        if item.kind == 'example_gradients' and not entered.is_set():
            entered.set()
            if not release.wait(10):
                raise RuntimeError('test timed out waiting for writer')
        return original(self, writer, item)

    monkeypatch.setattr(Recorder, '_handle', slow_first_trace)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    r = archx.attach(model, optimizer, str(tmp_path / 'bounded.archx.mdl'),
                     full_history=True, queue_size=1)
    errors = []

    def step():
        optimizer.zero_grad()
        parts = model(torch.tensor([[1.], [2.]])).reshape(-1) / 2
        parts.sum().backward(retain_graph=True)
        r.trace_examples(parts, [b'a', b'b'])
        r.log(data=b'ab')
        optimizer.step()

    step()
    assert entered.wait(10)

    def second():
        try:
            step()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=second)
    thread.start()
    try:
        assert not finished.wait(.2), 'trace should wait for the blocked writer'
    finally:
        release.set()
    thread.join(10)
    assert finished.is_set() and not errors
    r.close()
    assert [step for step, _, _ in read_example_gradients(Reader(tmp_path / 'bounded.archx.mdl'))] == [1, 2]
    assert r.stats.waited >= 1
