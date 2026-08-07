# SurvNG repository instructions

## Test execution

- Always run Python tests through `scripts/run-tests.sh`; never invoke
  `.venv/bin/pytest` or `python -m pytest` directly.
- Pass focused test paths and pytest options to the wrapper, for example
  `scripts/run-tests.sh -q tests/test_process_memory.py`.
- The wrapper owns a bounded process group and cleans up native inference
  children if a test run wedges or is interrupted.
- Test campaigns are serialized by the wrapper; do not bypass its lock by
  launching direct or parallel pytest processes.
