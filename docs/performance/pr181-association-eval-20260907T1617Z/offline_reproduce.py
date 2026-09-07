"""Portable replay-only wrapper; run from an isolated checkout at the manifest SHA.
Usage: python offline_reproduce.py PRIVATE_INPUT_DIR NEW_OUTPUT_DIR
Raw capture-EVENT.replay.json files must be supplied privately, not from the ZIP.
"""
import sys,json,subprocess,os
from pathlib import Path
source=Path(sys.argv[1]).resolve();dest=Path(sys.argv[2]).resolve()
dest.mkdir(mode=0o700,parents=True,exist_ok=False)
events=(63156,63346,64897,65548,64911,64006)
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','XDG_CACHE_HOME':str(dest),'TMPDIR':str(dest)}
for event in events:
    replay=source/f'capture-{event}.replay.json'
    if not replay.is_file():raise SystemExit(f'Missing private replay for event {event}; do not fabricate inputs')
    for profile in ('fixed_2fps','fixed_075fps')+(('sparse_gaps',) if event==64897 else ()):
        output=dest/f'{event}-{profile}.json'
        subprocess.run([sys.executable,'-m','survng.app.tracking_evaluation',str(replay),'--association-cues','--profile',profile,'--output',str(output)],env=env,check=True,timeout=90)
        result=json.loads(output.read_text())
        assert set(result['engines'])=={'survng_hybrid','survng_hybrid_multicue'}
        assert all(not e.get('error') for e in result['engines'].values())
