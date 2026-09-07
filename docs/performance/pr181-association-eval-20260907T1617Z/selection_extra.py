"""Six additional sparse selection frames from already resolved recording paths."""
import json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
for item in json.loads((ROOT/'private/media-candidates.json').read_text()):
    if item['slot'] not in ('moving_vehicle','small_distant'):continue
    sheet=Image.new('RGB',(1280,1140),'#111111');draw=ImageDraw.Draw(sheet)
    for i,offset in enumerate((4,10,22)):
        target=item['epoch']+offset;s=next(s for s in item['segments'] if s['start_epoch']<=target<s['end_epoch'])
        dest=ROOT/'private'/f'extra-{item["event_id"]}-{offset}.jpg'
        run=subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-ss',str(target-s['start_epoch']),'-i',s['path'],'-frames:v','1','-vf','scale=1280:340:force_original_aspect_ratio=decrease','-threads','1',str(dest)],capture_output=True,timeout=15)
        if run.returncode:raise RuntimeError('selection frame decode failed')
        sheet.paste(Image.open(dest),(0,i*380+30));draw.text((5,i*380+5),f'{item["camera_id"]} event {item["event_id"]} +{offset}s',fill='white')
    sheet.save(ROOT/'private'/f'extra-selection-{item["event_id"]}.jpg')
