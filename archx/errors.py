"""Exceptions. One class per failure kind, so callers can degrade selectively."""

from __future__ import annotations


class ArchxError(Exception):
    """Base class for every error archx raises deliberately."""


class FormatError(ArchxError):
    """The file is not a valid archx container, or its metadata contradicts its blocks."""


class CorruptBlockError(FormatError):
    """A block header or payload failed validation (CRC, length, or codec id)."""


class TruncatedFileError(FormatError):
    """The file ends inside a block. Recoverable with `archx recover`."""


class UnsupportedVersionError(FormatError):
    """The file's format major version is newer than this build understands."""


class DependencyMissingError(ArchxError):
    """An optional dependency is required for the requested operation."""
