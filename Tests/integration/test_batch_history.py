"""Default batch history must survive repeated data, missing data and joint updates."""
import json
import numpy as np
import pytest

torch = pytest.importorskip('torch')
import archx
from archx.cli.main import main
from archx.cli.history import get_history
from archx.container.reader import Reader
from archx.container.values import Timeline
from archx.data.batch import encode_batch, decode_batch


def test_default_every_step_has_exact_batch_and_weight_changes(tmp_path, capsys):
    torch.manual_seed(123)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    path = tmp_path / 'history.archx.mdl'
    r = archx.attach(model, optimizer, str(path))
    assert r.full_history
    a = {'tokens': torch.tensor([[1., 2.], [3., 4.]]), 'targets': torch.tensor([0, 1])}
    b = {'tokens': torch.tensor([[7., 8.], [9., 10.]]), 'targets': torch.tensor([1, 0])}
    expected = []
    for batches in ([a], [b], [a, b], []):
        before = model.weight.detach().clone()
        optimizer.zero_grad()
        actual = batches or [a]  # final step is intentionally not logged
        for batch in actual:
            loss = torch.nn.functional.cross_entropy(model(batch['tokens']), batch['targets']) / len(actual)
            loss.backward()
            if batches:
                r.log(loss=loss.item(), data=batch, source='dataset-A',
                      segment_id='reused-user-name', loss_reduction='mean')
        optimizer.step()
        expected.append((model.weight.detach() - before).numpy().copy())
    r.close()
    reader = Reader(path)
    assert reader.meta()['compression_level'] == 9
    assert Timeline(reader).steps == [0, 1, 2, 3, 4]
    report = get_history(path, tensor='weight', row=0, include_data=True)
    assert report['complete_recorded_range'] and report['file_closed']
    assert report['all_step_events_present'] and not report['all_recorded_batches_available']
    assert [x['step'] for x in report['changes']] == [4, 3, 2, 1]
    for change in report['changes']:
        assert change['delta_norm'] == pytest.approx(np.linalg.norm(expected[change['step'] - 1][0]), abs=1e-7)
    changes = {x['step']: x for x in report['changes']}
    assert changes[4]['batch_status'] == 'not_provided'
    assert len(changes[3]['batches']) == 2
    assert changes[1]['batches'][0]['hash'] == changes[3]['batches'][0]['hash']
    assert changes[2]['batches'][0]['hash'] == changes[3]['batches'][1]['hash']
    assert changes[1]['batches'][0]['hash'] != changes[2]['batches'][0]['hash']
    for step, batch in ((1, a), (2, b)):
        got = changes[step]['batches'][0]
        assert got['source'] == 'dataset-A' and got['loss_reduction'] == 'mean'
        assert got['content'] == 'stored'
        assert got['data']['tokens']['values'] == batch['tokens'].tolist()
        assert got['data']['targets']['values'] == batch['targets'].tolist()
    assert main(['history', str(path), '--tensor', 'weight', '--index', '0,0', '--json', '--data']) == 0
    cli = json.loads(capsys.readouterr().out)
    assert cli['tensor'] == 'weight' and cli['changes'][0]['step'] == 4
    assert 'not per-example causation' in cli['warning']
    assert main(['history', str(path), '--tensor', 'weight', '--index', '10,0']) == 1
    assert 'archx:' in capsys.readouterr().err


def test_pointer_only_and_explicit_adaptive_do_not_promise_data(tmp_path):
    m = torch.nn.Linear(1, 1); o = torch.optim.SGD(m.parameters(), lr=.1)
    path = tmp_path / 'pointer.archx.mdl'
    r = archx.attach(m, o, str(path), data_level='pointers')
    m(torch.ones((1, 1))).sum().backward(); r.log(data={'x': torch.ones((1, 1))})
    o.step(); r.close()
    h = get_history(path, tensor='weight', include_data=True)
    assert h['changes'][0]['batches'][0]['content'] == 'unavailable'
    m = torch.nn.Linear(1, 1); o = torch.optim.SGD(m.parameters(), lr=.1)
    r = archx.attach(m, o, str(tmp_path / 'adaptive.archx.mdl'), full_history=False)
    m(torch.ones((1, 1))).sum().backward(); o.step(); r.close()
    with pytest.raises(ValueError, match='adaptive'):
        get_history(tmp_path / 'adaptive.archx.mdl', tensor='weight')


def test_batch_roundtrip_and_reject_unsupported():
    data = {'text': 'ہیلو 🧠', 'bytes': b'\x00\xff', 'array': np.arange(12, dtype=np.int64).reshape(3, 4),
            'nested': (1, [True, None])}
    result = decode_batch(encode_batch(data))
    assert result['text'] == data['text'] and result['bytes'] == data['bytes']
    assert np.array_equal(result['array'], data['array'])
    assert result['nested'] == data['nested']
    with pytest.raises(ValueError, match='unsupported'):
        encode_batch({'bad': object()})


def test_data_cap_is_visible_and_cochanges_are_real(tmp_path):
    m = torch.nn.Linear(2, 1)
    o = torch.optim.SGD(m.parameters(), lr=.1)
    path = tmp_path / 'cap.archx.mdl'
    r = archx.attach(m, o, str(path), data_max_bytes=1)
    m(torch.ones((1, 2))).sum().backward()
    r.log(data={'input': torch.ones((1, 2))}, loss_reduction='sum')
    o.step(); r.close()
    h = get_history(path, tensor='weight', index=[0, 0], include_data=True, cochanges=2)
    assert h['file_closed'] and h['all_step_events_present']
    assert not h['all_recorded_batches_available']
    assert h['changes'][0]['batches'][0]['content'] == 'unavailable'
    co = h['changes'][0]['cochanging_weights_sample']
    assert co and all(not (v['tensor'] == 'weight' and v['index'] == [0, 0]) for v in co)
    assert any(v['tensor'] == 'weight' and v['index'] == [0, 1] for v in co)
