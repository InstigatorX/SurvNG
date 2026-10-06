import json
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from survng.app.media_exports import MediaExportManager
from survng.app.story_export import render_storyline


@pytest.mark.skipif(not shutil.which('ffmpeg') or not shutil.which('ffprobe'), reason='FFmpeg required')
def test_actual_story_export_split_crop_gap_and_existing_lifecycle(tmp_path):
    camera_files = {}
    for camera, color in [('gate', 'red'), ('porch', 'blue')]:
        path = tmp_path/f'{camera}.mp4'
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','lavfi','-i',f'color=c={color}:s=320x180:r=25','-t','3','-c:v','libx264','-pix_fmt','yuv420p','-y',str(path)], check=True)
        camera_files[camera] = path
    class Recorder:
        def __init__(self): self.leases = []
        def recording_rows_between(self, camera, start, end, source, **kwargs):
            return [{'path': str(camera_files[camera]),'start_epoch': 100 if start<105 else 110,'end_epoch': 103 if start<105 else 113,'duration_seconds': 3}]
        def lease_recordings_for_playback(self, rows, **kwargs): self.leases.extend(rows)
    recorder = Recorder()
    owner = MediaExportManager(tmp_path/'storage', tmp_path/'db', lambda: recorder, lambda: shutil.which('ffmpeg'), lambda: 'cpu')
    crop = [{'at':0,'x':.3,'y':.5,'size':.6},{'at':2.9,'x':.7,'y':.5,'size':.6}]
    plan = {'duration':8, 'missing_coverage': [], 'shots':[
        {'kind':'video','offset':0,'duration':3,'start':100,'end':103,'views':[{'camera_id':'gate','camera_name':'Gate','crop':crop,'crop_offset':0},{'camera_id':'porch','camera_name':'Porch','crop':[],'crop_offset':0}]},
        {'kind':'gap','offset':3,'duration':2,'start':103,'end':110,'elapsed_seconds':7},
        {'kind':'video','offset':5,'duration':3,'start':110,'end':113,'views':[{'camera_id':'gate','camera_name':'Gate','crop':[],'crop_offset':0}]}]}
    public = owner.create({'kind':'storyline','camera_id':'storyline','source':'main','start_epoch':100,'end_epoch':113,'label':'Arrival','options':{'replay_plan':plan}})
    job = owner.store.get(public['id'])
    owner._execute(job, threading.Event())
    completed = owner.get(public['id'])
    assert completed['status'] == 'completed'
    assert completed['download_url'].endswith('/download')
    path, name = owner.output_path(public['id'])
    assert path.parent.name == 'storyline'
    info = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)]))
    assert float(info['format']['duration']) == pytest.approx(8, abs=.12)
    stream = info['streams'][0]
    assert stream['width'] == 1280 and stream['height'] == 720
    frame = subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-ss','1','-i',str(path),'-vf','crop=4:2:300:360','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
    assert frame[0] > frame[2]  # Gate on the left.
    right = subprocess.check_output(['ffmpeg','-hide_banner','-loglevel','error','-ss','1','-i',str(path),'-vf','crop=4:2:900:360','-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'])
    assert right[2] > right[0]  # Porch on the right.
    assert recorder.leases
    manifest = json.loads((owner.manifest_dir/f"{public['id']}.json").read_text())
    assert manifest['options']['replay_plan'] == plan
    assert manifest['gaps'][0]['elapsed_seconds'] == 7
    assert owner.set_protected(public['id'], True)['protected']
    assert owner.set_label(public['id'], 'Reviewed delivery')['label'] == 'Reviewed delivery'
    owner.set_protected(public['id'], False)
    owner.cancel_or_delete(public['id'])
    assert not path.exists()


def test_export_refuses_lost_coverage_and_cancellation(tmp_path):
    owner = MediaExportManager(tmp_path/'storage', tmp_path/'db', lambda: type('Recorder', (), {'recording_rows_between': lambda *a,**k: []})(), lambda: 'ffmpeg', lambda: 'cpu')
    plan = {'shots':[{'kind':'video','duration':2,'start':100,'end':102,'views':[{'camera_id':'gate'}]}]}
    job = {'id':'fixture','options':{'replay_plan':plan}}
    with pytest.raises(RuntimeError, match='coverage changed'):
        render_storyline(owner, job, tmp_path, threading.Event())
    cancelled = threading.Event(); cancelled.set()
    with pytest.raises(InterruptedError):
        render_storyline(owner, job, tmp_path, cancelled)
