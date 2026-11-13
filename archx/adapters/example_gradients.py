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
            metadata, arrays = unshuffle_trace_payload(block.payload)
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
            descriptors.append({'shape': list(value.shape), 'dtype': value.dtype.str})
            arrays[f'g{i}'] = np.frombuffer(shuffle(value.tobytes(order='C'), 4), dtype=np.uint8)
    examples = [base64.b64decode(x, validate=True) for x in metadata.pop('examples_base64')]
    arrays['examples'] = np.frombuffer(b''.join(examples), dtype=np.uint8)
    metadata.update(version=2, encoding='byte_shuffle_4', tensors=descriptors,
                    example_lengths=[len(x) for x in examples])
    arrays['metadata'] = np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
    result = io.BytesIO()
    np.savez(result, **arrays)
    return result.getvalue()


def unshuffle_trace_payload(payload: bytes):
    """Decode a v2 archive into the same metadata and named float arrays as v1."""
    from ..encoders.plane import unshuffle
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive['metadata'].tobytes())
        if metadata.get('version') != 2 or metadata.get('encoding') != 'byte_shuffle_4':
            raise ValueError('unsupported example gradient encoding')
        if len(metadata['tensors']) != len(metadata['names']):
            raise ValueError('incorrect gradient tensor descriptors')
        arrays = {}
        for i, (name, spec) in enumerate(zip(metadata['names'], metadata['tensors'])):
            dtype = np.dtype(spec['dtype'])
            if dtype.kind != 'f' or dtype.itemsize != 4:
                raise ValueError('invalid gradient dtype')
            raw = archive[f'g{i}'].tobytes()
            arrays[name] = np.frombuffer(unshuffle(raw, 4), dtype=dtype).reshape(spec['shape']).copy()
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
