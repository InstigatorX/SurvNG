"""SurvNG tests: isolate application startup before importing any test module."""
from ._isolated_runtime import isolate_runtime

isolate_runtime()
