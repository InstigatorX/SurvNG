"""Bounded read-only recording-index and sparse frame inspection; no camera connection."""
import json,sqlite3,time,subprocess,os
from pathlib import Path
from datetime import datetime
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent
def main():
    os.umask(0o077)
    inputs=[('crossing1',64898,'detail-upper'),('crossing2',63346,'detail-crossing2'),('occlusion',63156,'detail-occlusion'),('moving_vehicle',65548,'detail-vehicle'),('small_distant',64911,'detail-small'),('control',64006,'detail-control-reserve')]
    with sqlite3.connect('file:DEPLOYMENT/runtime/recording-index/recordings.sqlite3?mode=ro',uri=True,timeout=2) as db:
        db.row_factory=sqlite3.Row;db.execute('PRAGMA query_only=ON')
        metadata=[]
        for slot,eid,name in inputs:
            d=json.loads((ROOT/'private'/(name+'.json')).read_text());e=next(e for e in d['events'] if e['id']==eid);epoch=datetime.fromisoformat(e['created_at']).timestamp()
            deadline=time.monotonic()+3;db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
            segs=[dict(r) for r in db.execute("select path,start_epoch,end_epoch,playable,validated from recordings where camera_id=? and source='main' and end_epoch>? and start_epoch<? order by start_epoch limit 12",(d['camera_id'],epoch,epoch+30))]
            cursor=epoch;gaps=[]
            for s in segs:
                s['exists']=Path(s['path']).is_file()
                if not s['exists'] or not s['playable']:continue
                if s['start_epoch']>cursor+.1:gaps.append([cursor-epoch,s['start_epoch']-epoch])
                cursor=max(cursor,s['end_epoch'])
            if cursor<epoch+30-.1:gaps.append([cursor-epoch,30])
            item={'slot':slot,'event_id':eid,'incident_id':d['id'],'camera_id':d['camera_id'],'timestamp':e['created_at'],'epoch':epoch,'segments':segs,'gaps':gaps,'continuous_index_coverage':bool(segs) and not gaps,'detail_file':name+'.json','frames':[]}
            # Only twelve tiny selection thumbnails in total; 4 each for ambiguous categories.
            if slot in ('crossing1','crossing2','occlusion') and not gaps:
                sheet=Image.new('RGB',(1280,760),'#111111');draw=ImageDraw.Draw(sheet)
                for i,offset in enumerate((0,6,12,23)):
                    target=epoch+offset;s=next((s for s in segs if s['start_epoch']<=target<s['end_epoch'] and s['exists']),None)
                    if not s:continue
                    dest=ROOT/'private'/f'select-{eid}-{offset}.jpg'
                    cmd=['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-threads','1','-ss',str(target-s['start_epoch']),'-i',s['path'],'-frames:v','1','-vf','scale=640:340:force_original_aspect_ratio=decrease','-threads','1',str(dest)]
                    run=subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=15)
                    item['frames'].append({'offset':offset,'returncode':run.returncode,'mapping':'segment start_epoch + seek offset; approximate visual selection only'})
                    if run.returncode==0 and dest.exists():
                        im=Image.open(dest);x=(i%2)*640;y=(i//2)*380;sheet.paste(im,(x,y+30));draw.text((x+5,y+5),f'{d["camera_id"]} event {eid} +{offset}s',fill='white')
                sheet.save(ROOT/'private'/f'selection-{eid}.jpg')
            metadata.append(item)
        (ROOT/'private/media-candidates.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps([{k:x[k] for k in ('slot','event_id','camera_id','continuous_index_coverage','gaps','frames')} for x in metadata],indent=2))
if __name__=='__main__':main()
