"""GitHub Actions: verify the default batch-linked history on a 1M model."""
import contextlib
import io
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import archx
from Bench.scale_1m_parameters import build_model, count_parameters, make_data
from archx.container.reader import Reader
from archx.container.values import Timeline
from archx.cli.main import main


def run():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    model = build_model(torch)
    params = count_parameters(model)
    assert 990_000 <= params <= 1_010_000
    optimizer = torch.optim.SGD(model.parameters(), lr=.02)
    batches = make_data(torch, steps=6, batch=2, seq=12, seed=71)
    path = Path('Bench/out/batch-history-1m.archx.mdl')
    path.parent.mkdir(parents=True, exist_ok=True)
    recorder = archx.attach(model, optimizer, str(path))  # exercise default, no flags
    states = [{n: p.detach().numpy().copy() for n, p in model.named_parameters()}]
    started = time.perf_counter()
    losses = []
    for tokens, targets in batches:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(tokens), targets)
        loss.backward()
        recorder.log(loss=loss.item(), data={'tokens': tokens, 'targets': targets},
                     source='synthetic-first-token', loss_reduction='mean')
        optimizer.step()
        losses.append(loss.item())
        states.append({n: p.detach().numpy().copy() for n, p in model.named_parameters()})
    recorder.close()
    elapsed = time.perf_counter() - started
    reader = Reader(path)
    assert reader.header.closed and reader.container.footer is not None
    timeline = Timeline(reader)
    assert timeline.steps == list(range(7))
    assert reader.meta()['weight_history'] == 'every_optimizer_step'
    assert reader.meta()['compression_level'] == 9
    worst = 0.
    changed = 0
    for step, deltas in timeline.deltas:
        for name, delta in deltas.items():
            expected = states[step][name] - states[step-1][name]
            assert np.array_equal(delta, expected), (step, name)
            changed += int(np.count_nonzero(delta))
        rebuilt = timeline.reconstruct(step)
        for name in states[step]:
            worst = max(worst, float(np.max(np.abs(rebuilt[name] - states[step][name]))))
    assert worst < 1e-6
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        status = main(['history', str(path), '--tensor', 'head.bias', '--data', '--json', '--cochanges', '3'])
    assert status == 0
    report = json.loads(buf.getvalue())
    assert report['complete_recorded_range'] and report['file_closed'] and report['all_step_events_present']
    assert report['all_recorded_batches_available']
    assert [c['step'] for c in report['changes']] == [6, 5, 4, 3, 2, 1]
    for change in report['changes']:
        step = change['step']
        assert change['cochanging_weights_sample']
        batch = change['batches'][0]
        assert batch['loss_reduction'] == 'mean' and batch['content'] == 'stored'
        assert batch['data']['tokens']['values'] == batches[step-1][0].tolist()
        assert batch['data']['targets']['values'] == batches[step-1][1].tolist()
    result = {'parameters': params, 'updates': 6, 'batch_size': 2, 'sequence_length': 12,
              'seconds_training_recording_and_close': elapsed, 'recording_bytes': path.stat().st_size,
              'raw_float32_anchor_and_deltas_bytes': params * 4 * 7,
              'changed_delta_values': changed, 'max_reconstruction_abs_error': worst,
              'losses': losses, 'cli_history_verified_steps': 6,
              'data': 'synthetic tokens and targets, mean loss, CPU SGD'}
    out = Path('Actions/out/scale/batch-history-Linux.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
