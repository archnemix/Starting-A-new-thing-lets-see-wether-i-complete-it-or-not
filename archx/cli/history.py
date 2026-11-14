"""History of one weight or tensor row, linked to the batches recorded at each update."""
from __future__ import annotations
import base64
import json
import sys
import zlib
import numpy as np
from ..container.reader import Reader
from ..container.schema import BlockType
from ..container.values import Timeline
from ..data.batch import decode_batch
from ..data.codecs import EncodedShard, decode_shard, content_id
from ..errors import ArchxError


def _json_value(value):
    if isinstance(value, np.ndarray):
        return {'dtype': str(value.dtype), 'shape': list(value.shape), 'values': value.tolist()}
    if isinstance(value, bytes):
        return {'bytes_base64': base64.b64encode(value).decode('ascii')}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    return value


def get_history(path, *, tensor, index=None, row=None, include_data=False, cochanges=0):
    if cochanges < 0 or cochanges > 10000:
        raise ValueError("cochanges must be between 0 and 10000")
    reader = Reader(path)
    timeline = Timeline(reader)
    infos = {info.key: info for info in timeline.tensors}
    if tensor not in infos:
        raise ValueError(f'tensor {tensor!r} not found in recording')
    info = infos[tensor]
    key = tensor
    if info.tie_group:
        key = next(x.key for x in timeline.tensors if x.tie_group == info.tie_group)
    if index is not None and row is not None:
        raise ValueError('choose either index or row')
    if index is not None:
        if len(index) != len(info.shape) or any(i < 0 or i >= n for i, n in zip(index, info.shape)):
            raise ValueError(f'index {index} does not match shape {info.shape}')
        selector = tuple(index)
    elif row is not None:
        if not info.shape or row < 0 or row >= info.shape[0]:
            raise ValueError(f'row {row} is outside shape {info.shape}')
        selector = row
    else:
        selector = (...,)
    if reader.meta().get('weight_history') != 'every_optimizer_step':
        raise ValueError('this file used adaptive recording; per-update batch history is unavailable')
    events = {e['step']: e for e in timeline.events()}
    shards = {}
    for block in reader.blocks(BlockType.DATA_SHARD):
        head, _, body = block.payload.partition(b'\n')
        meta = json.loads(head)
        shards.setdefault(meta['shard_id'], (meta, body))
    records = []
    for step, deltas in timeline.deltas:
        if key not in deltas:
            continue
        selected = np.asarray(deltas[key][selector])
        nonzero = int(np.count_nonzero(selected))
        if not nonzero:
            continue
        event = events.get(step, {})
        batches = []
        for ref in event.get('batches', []):
            item = dict(ref)
            pair = shards.get(ref['hash'])
            item['content'] = 'stored' if pair else 'unavailable'
            if include_data and pair:
                meta, body = pair
                shard = EncodedShard(meta['shard_id'], meta['codec'], body, meta['raw_len'],
                                     meta['content_crc32'], meta['reconstruction_error'])
                raw = decode_shard(shard)
                if len(raw) != meta['raw_len'] or (zlib.crc32(raw) & 0xffffffff) != meta['content_crc32'] or content_id(raw) != ref['hash']:
                    raise ValueError('batch content failed its integrity check')
                if ref.get('format') == 'structured_batch_v1':
                    item['data'] = _json_value(decode_batch(raw))
                else:
                    item['data'] = {'bytes_base64': base64.b64encode(raw).decode('ascii')}
            batches.append(item)
        other_weights = []
        if cochanges:
            for name, delta in deltas.items():
                indices = np.flatnonzero(delta)
                for flat in indices:
                    position = tuple(int(i) for i in np.unravel_index(flat, delta.shape))
                    if name == key and index is not None and position == tuple(index):
                        continue
                    other_weights.append({'tensor': name, 'index': list(position),
                                          'delta': float(delta[position])})
                    if len(other_weights) >= cochanges:
                        break
                if len(other_weights) >= cochanges:
                    break
        records.append({'step': step, 'cochanging_weights_sample': other_weights, 'changed_values': nonzero,
                        'delta_norm': float(np.linalg.norm(selected.astype(np.float64))),
                        'delta': float(selected) if selected.ndim == 0 else None,
                        'other_changed_tensors': sorted(k for k, v in deltas.items()
                                                        if k != key and np.any(v != 0)),
                        'batch_status': event.get('batch_status', 'not_recorded_by_this_version'),
                        'batches': batches})
    recorded = sorted(step for step, _ in timeline.deltas)
    complete = bool(0 in timeline.anchors) and recorded == list(range(1, (max(recorded) if recorded else 0) + 1))
    closed = reader.header.closed and reader.container.footer is not None and not reader.container.torn
    event_steps = set(events)
    all_events = all(step in event_steps for step in recorded)
    batches_complete = all(events.get(step, {}).get('batch_status') == 'provided' and
                           all(ref['hash'] in shards for ref in events[step].get('batches', []))
                           for step in recorded)
    return {'tensor': tensor, 'stored_as': key, 'shape': list(info.shape),
            'selection': {'index': index, 'row': row}, 'complete_recorded_range': complete,
            'file_closed': closed, 'all_step_events_present': all_events,
            'all_recorded_batches_available': batches_complete,
            'recorded_updates': len(recorded),
            'warning': 'Batches are associated with the joint optimizer update; this is not per-example causation. '
                       'A tensor row is not automatically a neuron.',
            'changes': list(reversed(records))}


def run_history(path, *, tensor, index=None, row=None, include_data=False, cochanges=0, as_json=False):
    try:
        parsed = None if index is None else [int(x) for x in index.split(',')]
        report = get_history(path, tensor=tensor, index=parsed, row=row, include_data=include_data, cochanges=cochanges)
    except (ArchxError, ValueError, OSError, KeyError) as exc:
        print(f'archx: cannot show history: {exc}', file=sys.stderr)
        return 1
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"tensor {tensor} shape {report['shape']} (newest first)")
        print(report['warning'])
        print(f"closed={report['file_closed']} events_complete={report['all_step_events_present']} batches_available={report['all_recorded_batches_available']}")
        for change in report['changes']:
            print(f"step {change['step']}: {change['changed_values']} changed, "
                  f"delta norm {change['delta_norm']:.6g}, batch {change['batch_status']}")
            for other in change['cochanging_weights_sample']:
                print(f"  cochange {other['tensor']} {other['index']}: {other['delta']:.6g}")
            for batch in change['batches']:
                print(f"  {batch['hash']} source={batch['source'] or '(not provided)'} "
                      f"content={batch['content']} reduction={batch['loss_reduction']}")
                if 'data' in batch:
                    print('  data: ' + json.dumps(batch['data'], ensure_ascii=False))
        if not report['complete_recorded_range']:
            print('warning: recorded step range has gaps')
    return 0
