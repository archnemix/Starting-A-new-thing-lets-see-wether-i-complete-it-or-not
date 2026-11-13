"""Lossless, opt-in per-example gradients from the *actual batched graph*.

These are local derivatives of additive loss terms, not causal attributions to model
behaviour or allocations of nonlinear optimizer updates. Computing every derivative is
expensive; this module refuses an estimated per-step size over the caller's budget.
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np


def encode_example_gradients(loss_parts, examples, parameters, *, max_bytes):
    import torch

    if loss_parts.ndim != 1 or not loss_parts.numel():
        raise ValueError("loss_parts must be a non-empty one-dimensional tensor")
    n = loss_parts.numel()
    if len(examples) != n or any(not isinstance(x, bytes) for x in examples):
        raise ValueError("provide one bytes example per loss part, including target if needed")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    named, seen = [], set()
    for name, p in parameters.items():
        if p.requires_grad and id(p) not in seen:
            if p.dtype != torch.float32:
                raise ValueError("example tracing currently supports float32 parameters only")
            seen.add(id(p))
            named.append((name, p))
    if not named:
        raise ValueError("no trainable parameters")
    estimate = sum(p.numel() * p.element_size() * n for _, p in named) + sum(map(len, examples))
    if estimate > max_bytes:
        raise ValueError(f"example trace requires at least {estimate} raw bytes; budget is {max_bytes}")
    rows = [[] for _ in named]
    for i in range(n):
        grads = torch.autograd.grad(loss_parts[i], [p for _, p in named],
                                    retain_graph=True, allow_unused=True)
        for j, ((_, p), g) in enumerate(zip(named, grads)):
            if g is not None and g.is_sparse:
                raise ValueError("sparse example gradients are not supported")
            value = torch.zeros_like(p) if g is None else g.detach()
            if not torch.isfinite(value).all():
                raise ValueError("non-finite per-example gradient")
            rows[j].append(value.cpu().numpy().copy())
    arrays = {}
    max_error = 0.0
    for j, (name, p) in enumerate(named):
        value = np.stack(rows[j])
        summed = torch.from_numpy(value).to(p.device).sum(dim=0)
        target = torch.zeros_like(p) if p.grad is None else p.grad.detach()
        if target.is_sparse or not torch.isfinite(target).all():
            raise ValueError(f"unsupported or non-finite aggregate gradient: {name}")
        error = (summed - target).abs().max().item() if target.numel() else 0.0
        max_error = max(max_error, error)
        if not torch.allclose(summed, target, atol=1e-6, rtol=1e-4):
            raise ValueError(f"per-example gradients do not sum to .grad for {name} "
                             f"(max error {error:.3g}); check scaling, clipping or accumulation")
        arrays[f"g{j}"] = value
    metadata = {
        "version": 1, "meaning": "gradient of each supplied additive loss part, not causal influence or optimizer update",
        "names": [name for name, _ in named],
        "examples_base64": [base64.b64encode(x).decode('ascii') for x in examples],
        "loss_parts": loss_parts.detach().cpu().tolist(), "max_aggregate_abs_error": max_error,
        "raw_gradient_bytes": sum(a.nbytes for a in arrays.values()),
    }
    arrays["metadata"] = np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
    output = io.BytesIO()
    # The container compresses the resulting block using its normal configured codec.
    np.savez(output, **arrays)
    return output.getvalue(), {"examples": n, "raw_gradient_bytes": metadata["raw_gradient_bytes"],
                               "max_aggregate_abs_error": max_error}


def read_example_gradients(reader, *, step=None):
    """Yield (step, metadata, named arrays) from the file; no training graph required."""
    from ..container.schema import BlockType
    for block in reader.blocks():
        if block.block_type not in (BlockType.EXAMPLE_GRADIENTS,
                                    BlockType.EXAMPLE_GRADIENTS_SHUFFLED):
            continue
        if step is not None and block.step != step:
            continue
        if block.block_type == BlockType.EXAMPLE_GRADIENTS_SHUFFLED:
            metadata, arrays = unshuffle_trace_payload(block.payload, max_output_bytes=reader.max_block_bytes)
        else:
            with np.load(io.BytesIO(block.payload), allow_pickle=False) as archive:
                metadata = json.loads(archive["metadata"].tobytes())
                if metadata.get('version') != 1:
                    raise ValueError('unsupported legacy example gradient encoding')
                arrays = {name: archive[f"g{i}"].copy() for i, name in enumerate(metadata["names"])}
        yield block.step, metadata, arrays



def shuffle_trace_payload(payload: bytes) -> bytes:
    """Create a reversible alternative archive; the writer chooses by compressed size."""
    from ..encoders.plane import shuffle
    from ..container.compress import compress, effective_codec
    from ..container.schema import Codec
    codec = effective_codec(Codec.ZSTD)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].tobytes())
        if metadata['version'] != 1:
            raise ValueError('only version 1 traces can be transformed')
        arrays = {}
        descriptors = []
        for i, _ in enumerate(metadata['names']):
            value = archive[f'g{i}']
            if value.dtype != np.dtype('float32'):
                raise ValueError('only float32 gradients can be transformed')
            raw = value.tobytes(order='C')
            shuffled = shuffle(raw, 4)
            options = [('raw', raw, {}), ('shuffle', shuffled, {})]
            # The bit test, rather than float != 0, preserves negative zero and all
            # other bit patterns. Sparse indices are delta-encoded little-endian u32.
            flat_bytes = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
            nonzero = np.flatnonzero(np.any(flat_bytes != 0, axis=1))
            if len(nonzero) * 2 < len(flat_bytes) and len(flat_bytes) < 2**32:
                indices = nonzero.astype(np.uint32)
                gaps = np.diff(indices, prepend=np.uint32(0)).astype('<u4', copy=False)
                sparse = gaps.tobytes() + flat_bytes[nonzero].tobytes()
                options.append(('sparse', sparse, {'nnz': len(nonzero)}))
            kind, packed, extra = min(options, key=lambda item: len(compress(item[1], codec)))
            descriptors.append({'shape': list(value.shape), 'dtype': value.dtype.str,
                                'kind': kind, **extra})
            arrays[f'g{i}'] = np.frombuffer(packed, dtype=np.uint8)
    examples = [base64.b64decode(x, validate=True) for x in metadata.pop('examples_base64')]
    arrays['examples'] = np.frombuffer(b''.join(examples), dtype=np.uint8)
    metadata.update(version=2, encoding='adaptive_float32_v1', tensors=descriptors,
                    example_lengths=[len(x) for x in examples])
    arrays['metadata'] = np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
    result = io.BytesIO()
    np.savez(result, **arrays)
    return result.getvalue()


def unshuffle_trace_payload(payload: bytes, *, max_output_bytes=2 * 1024**3):
    """Decode a v2 archive into the same metadata and named float arrays as v1."""
    from ..encoders.plane import unshuffle
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].tobytes())
        if metadata.get('version') != 2 or metadata.get('encoding') not in (
                'byte_shuffle_4', 'selective_byte_shuffle_4', 'adaptive_float32_v1'):
            raise ValueError('unsupported example gradient encoding')
        if len(metadata['tensors']) != len(metadata['names']):
            raise ValueError('incorrect gradient tensor descriptors')
        arrays = {}
        output_size = 0
        import math
        for i, (name, spec) in enumerate(zip(metadata['names'], metadata['tensors'])):
            shape = spec['shape']
            if (not isinstance(shape, list) or len(shape) > 64 or
                    any(not isinstance(d, int) or d < 0 for d in shape)):
                raise ValueError('invalid gradient shape')
            count = math.prod(shape)
            output_size += count * 4
            if output_size > max_output_bytes:
                raise ValueError('gradient trace exceeds decoded size limit')
            if name in arrays:
                raise ValueError('duplicate gradient name')
            dtype = np.dtype(spec['dtype'])
            if dtype.kind != 'f' or dtype.itemsize != 4:
                raise ValueError('invalid gradient dtype')
            raw = archive[f'g{i}'].tobytes()
            if metadata['encoding'] == 'adaptive_float32_v1':
                kind = spec.get('kind')
                if kind == 'shuffle':
                    raw = unshuffle(raw, 4)
                elif kind == 'sparse':
                    nnz = spec.get('nnz')
                    if not isinstance(nnz, int) or nnz < 0 or nnz > count or len(raw) != nnz * 8:
                        raise ValueError('invalid sparse gradient')
                    gaps = np.frombuffer(raw[:nnz * 4], dtype='<u4').astype(np.uint64)
                    indices = np.cumsum(gaps, dtype=np.uint64)
                    if nnz and (indices[-1] >= count or np.any(gaps[1:] == 0)):
                        raise ValueError('invalid sparse gradient indices')
                    restored = np.zeros((count, 4), dtype=np.uint8)
                    restored[indices.astype(np.intp)] = np.frombuffer(raw[nnz * 4:], dtype=np.uint8).reshape(-1, 4)
                    raw = restored.tobytes()
                elif kind != 'raw':
                    raise ValueError('invalid gradient representation')
            elif metadata['encoding'] == 'byte_shuffle_4' or spec.get('shuffled') is True:
                raw = unshuffle(raw, 4)
            elif spec.get('shuffled') is not False:
                raise ValueError('invalid shuffle flag')
            arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(spec['shape']).copy()
        data = archive['examples'].tobytes()
        lengths = metadata['example_lengths']
        if any(not isinstance(n, int) or n < 0 for n in lengths) or sum(lengths) != len(data):
            raise ValueError('invalid example lengths')
        examples = []
        offset = 0
        for length in lengths:
            examples.append(base64.b64encode(data[offset:offset + length]).decode('ascii'))
            offset += length
        metadata['examples_base64'] = examples
        return metadata, arrays


def legacy_trace_payload(metadata, arrays):
    """Construct the v1 representation of a decoded trace for comparison/export."""
    keys = ('meaning', 'names', 'examples_base64', 'loss_parts',
            'max_aggregate_abs_error', 'raw_gradient_bytes')
    legacy = {'version': 1, **{key: metadata[key] for key in keys}}
    values = {f'g{i}': arrays[name] for i, name in enumerate(legacy['names'])}
    values['metadata'] = np.frombuffer(json.dumps(legacy, allow_nan=False).encode(), dtype=np.uint8)
    result = io.BytesIO()
    np.savez(result, **values)
    return result.getvalue()
