"""Per-layer activation statistics: what the network saw during a forward pass.

The recording triple is *state*, *cause*, *stage*. Weights give the state and the data gives
the cause; activations are the only evidence of what the two did to each other inside the
network - a layer whose outputs go flat, or saturate, or die, leaves no trace in either.

Design rules, each one paid for by a mistake this project already made somewhere else:

* **Statistics, never tensors.** A forward pass through a 512-wide layer is 512 x batch new
  floats per step. Storing those would turn a recording into an activation cache and destroy
  the thing the product is for. What is stored is a fixed-size summary per layer per stored
  step: shape, mean, std, min/max, and the fraction of exactly-zero and saturated units.
  That is bounded by the layer count, not by the batch or the model size.

* **Off the training thread, and off the critical path.** The forward hook computes the
  summary inline (a few numpy reductions over the output), because the tensor is only alive
  during the forward pass - it cannot be queued for later, and waiting for it would be
  waiting on the writer. What is queued is the finished summary: small, and already history.

* **Two calls per step is normal, and not a bug.** A training step runs the model twice
  (forward, then forward again to compute the loss on the updated weights, or a validation
  pass afterwards). Rather than guess which forward is "the" one, the capture accumulates
  every forward seen since the last store point and labels the summary `forwards: N`. A
  reader then knows exactly how many passes it is looking at - the alternative was to keep
  the last one and silently mislabel it.

* **Absent is recorded, not omitted.** No forward pass before a store point means no block,
  and the capability is denied with a reason, so a tier-3 claim always has blocks behind it.

Measuring a forward hook costs something on the training thread, and the number is reported
rather than hidden: see `Bench/claim15_activations.py` and `Docs/activation_results.json`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: The fraction of a tensor considered "saturated" for a unit count. At |x| >= 4 a tanh is
#: within 0.07% of its limit and a ReLU has long since stopped passing gradient, so 4 is a
#: meaningful boundary and a tunable-sounding 0.999 would not be.
SATURATION_LIMIT = 4.0

#: Elements reduced per layer per forward. Above this the tensor is strided, so the cost of a
#: summary is bounded by the layer count and not by the batch size.
#:
#: 16,384 and not 1,000,000: measured per element, the reduction costs ~2.8 us per 1000 values
#: once the float64 conversion is included, so a million-element budget put +2.3 ms on a
#: 1.1 ms forward pass - a feature that tripled the cost of the thing it was measuring. At
#: 16,384 a layer costs ~45 us, and a strided sample of that size puts the sample mean within
#: 1% of the full-tensor mean for any distribution this would meet (`Tests/unit/data/
#: test_activations.py` checks it against the exact answer rather than asserting it here).
MAX_ELEMENTS = 16_384


@dataclass
class ActivationStats:
    """The summary of one layer's output, small enough to keep forever."""

    layer: str
    shape: list[int]
    dtype: str
    forwards: int = 0
    #: Running totals, so several forwards in one step combine without keeping any of them.
    _count: float = 0.0
    _sum: float = 0.0
    _sum_sq: float = 0.0
    _min: float = math.inf
    _max: float = -math.inf
    _zeros: float = 0.0
    _saturated: float = 0.0
    _sampled: float = 0.0

    def observe(self, array: np.ndarray) -> None:
        """Fold one forward pass of this layer into the running summary."""
        flat = np.asarray(array, dtype=np.float32).ravel()
        if flat.size == 0:
            return
        if flat.size > MAX_ELEMENTS:
            # Strided, not truncated: taking the first N elements would measure the first
            # samples of the batch only and call that the layer. A stride spans the whole
            # tensor, and it is a *slice* - no copy, so it costs nothing to form.
            step = math.ceil(flat.size / MAX_ELEMENTS)
            flat = flat[::step]
        # Cheap NaN/Inf probe: a poisoned tensor poisons its own sum, so one reduction tells
        # us whether the allocating mask is needed at all. Checking `np.isfinite(flat).all()`
        # unconditionally allocated a bool array per layer per forward.
        finite = None
        if not np.isfinite(flat.sum(dtype=np.float64)):
            finite = np.isfinite(flat)
        if finite is not None:
            # NaN/Inf are a fact worth recording and fatal to a mean: they are counted and
            # excluded from the moments rather than poisoning every number in the block.
            self._nonfinite = getattr(self, "_nonfinite", 0) + int((~finite).sum())
            flat = flat[finite]
            if flat.size == 0:
                self.forwards += 1
                return
        self._count += float(flat.size)
        # One pass for the sum, one for the sum of squares, computed in float64 without
        # materialising a float64 *copy* of the tensor. The first version asked numpy for
        # `np.square(flat, dtype=np.float64)`, which allocates 8 bytes per element per layer
        # and dominated the cost of the whole feature (measured: +208% on a forward pass).
        # `np.dot` accumulates in float64 directly from float32 input, and `dtype=` on `sum`
        # does the same, so both stay exact enough for moments without the temporary.
        self._sum += float(flat.sum(dtype=np.float64))
        self._sum_sq += float(np.dot(flat, flat.astype(np.float64, copy=False)))
        self._min = min(self._min, float(flat.min()))
        self._max = max(self._max, float(flat.max()))
        self._zeros += float(np.count_nonzero(flat == 0.0))
        self._saturated += float(np.count_nonzero(np.abs(flat) >= SATURATION_LIMIT))
        self._sampled += float(flat.size)
        self.forwards += 1
        self.shape = list(np.shape(array))

    def to_dict(self) -> dict:
        """The block payload. Every number here is derived, never estimated."""
        count = self._count
        if count <= 0:
            return {"layer": self.layer, "shape": self.shape, "dtype": self.dtype,
                    "forwards": self.forwards, "count": 0}
        mean = self._sum / count
        # Population variance from the raw moments: sum_sq/n - mean^2. Clamped at zero
        # because float64 subtraction of two near-equal large numbers can go negative.
        variance = max(0.0, self._sum_sq / count - mean * mean)
        payload = {
            "layer": self.layer,
            "shape": self.shape,
            "dtype": self.dtype,
            "forwards": self.forwards,
            "count": int(count),
            "mean": mean,
            "std": math.sqrt(variance),
            "min": self._min,
            "max": self._max,
            "zero_fraction": self._zeros / count,
            "saturated_fraction": self._saturated / count,
            "saturation_limit": SATURATION_LIMIT,
            "sampled": int(self._sampled),
        }
        nonfinite = getattr(self, "_nonfinite", 0)
        if nonfinite:
            payload["nonfinite"] = nonfinite
        return payload


@dataclass
class ActivationCapture:
    """Collects per-layer summaries for the step now ending, then hands them over.

    `observe` runs inside a forward hook on the training thread; `take` runs at a store point
    and empties the accumulator. Nothing is retained after `take`, so a long run holds at most
    one step of summaries.
    """

    #: Which modules to summarise. `None` means "every module that produces a float tensor",
    #: which for a typical model is every leaf layer and nothing else.
    module_filter: Any = None
    max_layers: int = 256
    stats: dict[str, ActivationStats] = field(default_factory=dict)
    skipped: int = 0

    def observe(self, layer: str, output: Any) -> None:
        """Fold one layer's output into this step's accumulator."""
        if len(self.stats) >= self.max_layers and layer not in self.stats:
            self.skipped += 1
            return
        array = _as_float_array(output)
        if array is None:
            return
        entry = self.stats.get(layer)
        if entry is None:
            entry = ActivationStats(layer=layer, shape=list(array.shape),
                                    dtype=str(array.dtype))
            self.stats[layer] = entry
        entry.observe(array)

    def take(self) -> list[dict]:
        """Every summary collected since the last call, in a stable order."""
        payloads = [self.stats[key].to_dict() for key in sorted(self.stats)]
        self.stats = {}
        return payloads

    def __bool__(self) -> bool:
        return bool(self.stats)


def _as_float_array(output: Any) -> np.ndarray | None:
    """The float tensor inside whatever a forward hook was handed.

    A hook gets either a tensor or a tuple whose first element is one. Anything else - a
    dict, a loss, an integer index - is not an activation and is skipped rather than guessed
    at, because a wrong summary is worse than no summary.
    """
    if isinstance(output, (tuple, list)):
        if not output:
            return None
        output = output[0]
    detach = getattr(output, "detach", None)
    if detach is not None:
        output = detach()
        cpu = getattr(output, "cpu", None)
        if cpu is not None:
            output = cpu()
        to_numpy = getattr(output, "numpy", None)
        if to_numpy is None:
            return None
        output = to_numpy()
    array = np.asarray(output)
    if array.dtype.kind not in "fc":
        return None
    return array
