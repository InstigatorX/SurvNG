#!/usr/bin/env python3
"""Single bounded py-spy capture; no service modification or Python injection."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / 'profiler-env/bin/py-spy'


def save(value):
    (ROOT / 'profile-window.json').write_text(json.dumps(value, indent=2) + '\n')


def main():
    help_text = subprocess.check_output([str(BINARY), 'record', '--help'], text=True, timeout=5)
    (ROOT / 'py-spy-record-help.txt').write_text(help_text)
    version = subprocess.check_output([str(BINARY), '--version'], text=True, timeout=5).strip()
    packages = subprocess.check_output([str(ROOT / 'profiler-env/bin/python'), '-m', 'pip', 'freeze'], text=True, timeout=5)
    (ROOT / 'profiler-packages.txt').write_text(packages)
    command = [str(BINARY), 'record', '--pid', '651962', '--rate', '25', '--duration', '60',
               '--format', 'speedscope', '--native', '--threads', '--full-filenames',
               '--output', str(ROOT / 'pyspy-native.speedscope.json')]
    started = time.monotonic()
    record = {'tool': version, 'binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
              'target_pid': 651962, 'command': command, 'requested_duration_seconds': 60,
              'requested_rate_hz': 25, 'native_requested': True,
              'gil_filter': False, 'include_idle': False, 'subprocesses': False,
              'capture_locals': False, 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(),
              'started_monotonic': started, 'status': 'running'}
    with (ROOT / 'pyspy-record.log').open('wb') as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        record['profiler_pid'] = process.pid
        save(record)
        try:
            record['returncode'] = process.wait(timeout=75)
            record['status'] = 'complete' if process.returncode == 0 else 'failed'
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            record['status'] = 'timeout'
        finally:
            record['ended_at'] = dt.datetime.now(dt.timezone.utc).isoformat()
            record['ended_monotonic'] = time.monotonic()
            record['elapsed_seconds'] = record['ended_monotonic'] - started
            save(record)


if __name__ == '__main__':
    main()
