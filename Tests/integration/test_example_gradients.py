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
    r = archx.attach(m, opt, str(tmp_path / 'adaptive.archx.mdl'))
    with pytest.raises(ValueError, match='full_history'):
        r.trace_examples(torch.ones(1), [b'a'])
    r.close()
