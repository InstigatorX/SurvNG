"""Isolate pytest collection as well as package-based unittest execution."""
from tests._isolated_runtime import isolate_runtime

isolate_runtime()
