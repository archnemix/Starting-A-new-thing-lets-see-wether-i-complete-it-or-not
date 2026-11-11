"""Version constants in one place, so the file format and the package never disagree."""

from __future__ import annotations

__version__ = "0.1.0"

#: Container format written by this build. Major.Minor, per Plan/Format_Spec.md.
FORMAT_MAJOR = 1
FORMAT_MINOR = 0

MAGIC = b"ARCHXMDL"
END_MAGIC = b"ARCHXEND"
