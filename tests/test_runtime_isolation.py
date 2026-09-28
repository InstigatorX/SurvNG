"""A non-pytest test import must never select the configured live database."""
import os
import subprocess
import sys
from pathlib import Path


def test_named_unittest_import_replaces_inherited_runtime_configuration():
    script = '''
import json, os
from pathlib import Path
import tests
config_path=Path(os.environ['SURVNG_CONFIG_PATH'])
assert config_path != Path('/must-not-be-used/config.json')
config=json.loads(config_path.read_text())
assert config['cameras']==[]
assert config['retention']['enabled'] is False
for name in ('storage_dir','database_dir','recording_index_dir'):
    assert Path(config[name]).parent==config_path.parent
from survng.app import main
assert os.environ['SURVNG_CONFIG_PATH']==str(config_path)
'''
    result=subprocess.run([sys.executable,'-c',script],cwd=Path(__file__).resolve().parents[1],
                          env={**os.environ,'SURVNG_CONFIG_PATH':'/must-not-be-used/config.json'},
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
