"""Canonical JSON and content addresses shared by every formal artifact.

A content address is ``sha256:`` followed by the hex digest of the exact bytes
it names. JSON objects are addressed through one canonical encoding (sorted
keys, no insignificant whitespace, UTF-8, no NaN), so equal objects always
have equal addresses on both sides of the contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

SHA256_PATTERN = r"^sha256:[0-9a-f]{64}$"
_SHA256 = re.compile(SHA256_PATTERN)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_ref(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def object_ref(value: Any) -> str:
    """The content address of ``value``'s canonical JSON encoding."""
    return sha256_ref(canonical_json(value))


def is_sha256_ref(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None
