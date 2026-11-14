"""Lossless structured batches: named arrays, inputs and targets with their boundaries.

The trainer must explicitly supply the data; this format does not guess its semantics.
"""
from __future__ import annotations
import base64
import io
import json
import numpy as np


def encode_batch(data):
    arrays = {}
    def encode(value):
        if hasattr(value, 'detach') and hasattr(value, 'cpu'):
            try:
                value = value.detach().cpu().numpy()
            except (TypeError, RuntimeError) as exc:
                raise ValueError('batch tensor dtype is not supported for exact capture') from exc
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise ValueError('object arrays cannot be captured safely')
            key = f'a{len(arrays)}'
            arrays[key] = np.ascontiguousarray(value)
            return {'type': 'array', 'key': key}
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            return {'type': 'bytes', 'value': base64.b64encode(value).decode('ascii')}
        if isinstance(value, str):
            return {'type': 'str', 'value': value}
        if value is None or isinstance(value, (bool, int, float)):
            return {'type': 'scalar', 'value': value}
        if isinstance(value, (list, tuple)):
            return {'type': 'tuple' if isinstance(value, tuple) else 'list',
                    'items': [encode(item) for item in value]}
        if isinstance(value, dict) and all(isinstance(k, str) for k in value):
            return {'type': 'dict', 'items': {k: encode(v) for k, v in value.items()}}
        raise ValueError(f'unsupported batch value {type(value).__name__}; supply tensors, text or bytes')
    tree = encode(data)
    arrays['manifest'] = np.frombuffer(json.dumps({'version': 1, 'tree': tree},
                                                  allow_nan=False).encode('utf-8'), dtype=np.uint8)
    result = io.BytesIO()
    np.savez(result, **arrays)
    return result.getvalue()


def decode_batch(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        manifest = json.loads(archive['manifest'].tobytes())
        if manifest.get('version') != 1:
            raise ValueError('unsupported batch version')
        def decode(node):
            kind = node['type']
            if kind == 'array':
                return archive[node['key']].copy()
            if kind == 'bytes':
                return base64.b64decode(node['value'], validate=True)
            if kind in ('str', 'scalar'):
                return node['value']
            if kind == 'dict':
                return {k: decode(v) for k, v in node['items'].items()}
            if kind in ('list', 'tuple'):
                result = [decode(v) for v in node['items']]
                return tuple(result) if kind == 'tuple' else result
            raise ValueError(f'unsupported batch node {kind!r}')
        return decode(manifest['tree'])
