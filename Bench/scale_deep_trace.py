"""GitHub Actions proof of maximum-detail recording on a 1,004,176-parameter model.

CPU, synthetic data, plain SGD. The per-example gradients here are additive components
of a batch loss, NOT evidence of an example's causal influence on future behaviour.
"""
from __future__ import annotations
import argparse
import base64
import json
import sys
import time
from pathlib import Path

# When invoked as a script, put the repo root on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import archx
from Bench.scale_1m_parameters import build_model, count_parameters, make_data
from archx.container.reader import Reader
from archx.container.schema import BlockType
from archx.container.values import Timeline
from archx.adapters.example_gradients import read_example_gradients


def run(steps, batch, output):
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    model = build_model(torch)
    count = count_parameters(model)
    assert 990_000 <= count <= 1_010_000, count
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    batches = make_data(torch, steps=steps, batch=batch, seq=12, seed=71)
    config = {'model': 'Bench.scale_1m_parameters.build_model', 'model_seed': 2026,
              'data_seed': 71, 'steps': steps, 'batch': batch, 'sequence_length': 12,
              'loss': 'cross_entropy mean', 'optimizer': 'SGD no momentum/weight decay',
              'learning_rate': 0.02, 'dataset': 'synthetic first-token class'}
    output.parent.mkdir(parents=True, exist_ok=True)
    recorder = archx.attach(model, optimizer, str(output), full_history=True, max_rel_error=0,
                            meta={'run_config': config})
    weights = [{n: p.detach().cpu().numpy().copy() for n, p in model.named_parameters()}]
    gradients, examples, losses, trace_times = [], [], [], []
    t0 = time.perf_counter()
    for tokens, labels in batches:
        optimizer.zero_grad()
        logits = model(tokens)
        parts = torch.nn.functional.cross_entropy(logits, labels, reduction='none') / batch
        parts.sum().backward(retain_graph=True)
        sample_bytes = [json.dumps({'tokens': x.tolist(), 'label': int(y)},
                                   separators=(',', ':')).encode()
                        for x, y in zip(tokens, labels)]
        t = time.perf_counter()
        recorder.trace_examples(parts, sample_bytes)
        trace_times.append(time.perf_counter() - t)
        gradients.append({n: p.grad.detach().cpu().numpy().copy()
                          for n, p in model.named_parameters()})
        examples.append(sample_bytes)
        losses.append(parts.sum().item())
        recorder.log(loss=losses[-1], data=tokens)
        optimizer.step()
        weights.append({n: p.detach().cpu().numpy().copy() for n, p in model.named_parameters()})
    recorder.close()
    duration = time.perf_counter() - t0
    reader = Reader(output)
    timeline = Timeline(reader)
    assert timeline.steps == list(range(steps + 1)), timeline.steps
    assert len(list(reader.blocks(BlockType.ANCHOR))) == 1
    assert len(list(reader.blocks(BlockType.DELTA))) == steps
    assert reader.meta()['run_config'] == config
    worst_weight = 0.0
    for step in timeline.steps:
        state = timeline.reconstruct(step)
        for name, live in weights[step].items():
            worst_weight = max(worst_weight, float(np.max(np.abs(state[name] - live))))
    traces = list(read_example_gradients(reader))
    assert [step for step, _, _ in traces] == list(range(1, steps + 1))
    worst_grad, worst_update, raw_trace_bytes = 0., 0., 0
    for step, meta, arrays in traces:
        assert [base64.b64decode(s) for s in meta['examples_base64']] == examples[step - 1]
        raw_trace_bytes += meta['raw_gradient_bytes']
        for name, arr in arrays.items():
            assert arr.shape[0] == batch
            g = arr.sum(axis=0)
            worst_grad = max(worst_grad, float(np.max(np.abs(g - gradients[step-1][name]))))
            actual = weights[step][name] - weights[step-1][name]
            worst_update = max(worst_update, float(np.max(np.abs(actual + .02 * g))))
    assert worst_weight < 1e-6, worst_weight
    assert worst_grad < 1e-5, worst_grad
    assert worst_update < 1e-6, worst_update
    trace_blocks = list(reader.blocks(BlockType.EXAMPLE_GRADIENTS))
    report = {'parameters': count, 'steps': steps, 'batch': batch,
              'initial_anchors': 1, 'deltas': steps, 'example_records': len(traces),
              'examples_checked': steps * batch, 'loss_first': losses[0], 'loss_last': losses[-1],
              'max_weight_error': worst_weight, 'max_gradient_sum_error': worst_grad,
              'max_sgd_update_error': worst_update, 'raw_gradient_bytes': raw_trace_bytes,
              'compressed_trace_block_bytes': sum(b.stored_len for b in trace_blocks),
              'recording_bytes': output.stat().st_size,
              'checkpoint_weights_bytes': count * 4,
              'trace_seconds_total': sum(trace_times), 'wall_seconds': duration,
              'meaning': 'local loss gradients; not causal effects or general optimizer attribution',
              'run_config': config}
    return report


def main():
    p = argparse.ArgumentParser(); p.add_argument('--steps', type=int, default=6)
    p.add_argument('--batch', type=int, default=2)
    p.add_argument('--out', type=Path, default=Path('Bench/out/deep-1m.archx.mdl'))
    p.add_argument('--json-out', type=Path, required=True)
    a = p.parse_args()
    if a.steps < 1 or a.batch < 1:
        p.error('steps and batch must be positive')
    report = run(a.steps, a.batch, a.out)
    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
