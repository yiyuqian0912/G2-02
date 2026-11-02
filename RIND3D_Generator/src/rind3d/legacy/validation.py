"""Explicit legacy record checks; never waives current publication gates."""
from ..pipeline import validate_records as _validate
from ..pipeline import write_candidate_files as _write


def validate_records(root,meta,*,strict=True):
    return _validate(root,meta,strict=strict,legacy_diagnostic=not strict)


def write_candidate_files(root,windows,bank,ids,reports):
    return _write(root,windows,bank,ids,reports,legacy_diagnostic=True)
