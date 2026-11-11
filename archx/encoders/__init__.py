"""Delta encoders. Each one turns a float delta into a smaller representation.

Every encoder reports what it stored and what it lost, so the file can record the price of
the choice instead of pretending the choice was free.
"""

from __future__ import annotations

from .base import Encoded, Encoder, choose_encoder
from .dense import DenseQuant
from .dense_raw import DenseRaw
from .lowrank import LowRank
from .sparse import Sparse
from .unchanged import Unchanged

__all__ = ["Encoded", "Encoder", "choose_encoder", "Unchanged", "Sparse", "DenseQuant",
           "LowRank", "DenseRaw"]
