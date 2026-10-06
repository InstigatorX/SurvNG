"""Physical scene-change measurements shared by acquisition and tracking."""
import hashlib
import json
import math

import cv2
import numpy as np

from .detector import detection_failure
from .scene_identity import observation_identity
from .scene_activity import LOCALIZED_CHANGE_POLICY_VERSION, localized_change_supported


def scene_sample_records(samples, observations, event_epoch, camera_id, confirmation_offsets=None, *, source="recorded_main"):
    """Describe native acquisition and measure localized physical change.

    The thumbnail calculation is bounded to 320 pixels wide. Original frames
    remain unchanged and are the only source of the returned review image.
    """
    def stable_id(prefix, value):
        return prefix+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()[:32]

    for observation in observations:
        observation["id"]=observation_identity(camera_id, observation["captured_at_epoch"], observation)
    records=[]
    available=[]
    for sample in sorted(samples,key=lambda item:item.offset):
        captured=round(event_epoch+sample.offset,6)
        good=sample.frame is not None and not detection_failure(sample.objects)
        height,width=sample.frame.shape[:2] if sample.frame is not None else (0,0)
        frame_observations=[o for o in observations if abs(o["captured_at_epoch"]-captured)<1e-6]
        # An analysis variant retains its own detections. The activity evaluator
        # still counts camera/capture time, never analysis IDs, as distinct frames.
        record={"id":stable_id("sample-",[camera_id,captured,source,sample.recording_path,width,height,good,
                                         sorted({o["id"] for o in frame_observations})]),
                "camera_id":camera_id,"captured_epoch":captured,"status":"complete" if good else "failed",
                "observations":frame_observations,
                "metadata":{"source":source,"recording_path":sample.recording_path,
                            "frame_width":width,"frame_height":height,"frame_timestamp_exact":sample.exact_timestamp,
                            "activity_witnesses":[],"confirmation_complete":confirmation_offsets is not None}}
        records.append(record)
        if good:
            scaled_width=min(320,width)
            small=cv2.resize(sample.frame,(scaled_width,max(1,round(height*scaled_width/width))))
            gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY) if small.ndim==3 else small
            gray=cv2.GaussianBlur(gray,(3,3),0).astype(np.float32)
            available.append((record,gray))
    for (previous,a),(current,b) in zip(available,available[1:]):
        elapsed=current["captured_epoch"]-previous["captured_epoch"]
        if not .05<=elapsed<=10 or a.shape!=b.shape:
            continue
        # A resolution/aspect change is not a validated cross-stream transform.
        if any(current["metadata"][key]!=previous["metadata"][key] for key in ("frame_width","frame_height")):
            continue
        residual=b-a
        residual-=np.median(residual)
        difference=np.abs(residual)
        threshold=max(12.0,6.0*float(np.median(difference)))
        changed=difference>threshold
        height,width=b.shape
        for observation in current["observations"]:
            box=observation["box"]
            fw=float(observation.get("detection_frame_width") or current["metadata"]["frame_width"])
            fh=float(observation.get("detection_frame_height") or current["metadata"]["frame_height"])
            coords=[float(box[key])/(fw if key.startswith("x") else fh) for key in ("x1","y1","x2","y2")]
            x1,y1,x2,y2=coords
            if not (0<=x1<x2<=1 and 0<=y1<y2<=1):
                continue
            center=np.array([(x1+x2)/2,(y1+y2)/2])
            # Match only a nearby source observation; object scores and alert
            # zones never contribute physical evidence.
            prior=[]
            for other in previous["observations"]:
                if other["label"]!=observation["label"]:
                    continue
                ob=other["box"]
                ow=float(other.get("detection_frame_width") or previous["metadata"]["frame_width"])
                oh=float(other.get("detection_frame_height") or previous["metadata"]["frame_height"])
                p=np.array([(ob["x1"]+ob["x2"])/(2*ow),(ob["y1"]+ob["y2"])/(2*oh)])
                distance=float(np.linalg.norm(center-p))
                if distance<=max(.15,math.hypot(x2-x1,y2-y1)):
                    prior.append((distance,p,other,ow,oh))
            prior.sort(key=lambda value:value[0])
            displacement=prior[0][0] if prior else 0.0
            # A stationary detection has no trajectory. It can still support
            # arrival if the actual image inside its ROI changed from baseline.
            kind=("localized_movement" if displacement>=.01 else "localized_motion") if prior else "localized_arrival"
            rx1,ry1,rx2,ry2=x1,y1,x2,y2
            if prior:
                old,ow,oh=prior[0][2:]
                ob=old["box"]
                rx1,ry1=min(rx1,ob["x1"]/ow),min(ry1,ob["y1"]/oh)
                rx2,ry2=max(rx2,ob["x2"]/ow),max(ry2,ob["y2"]/oh)
            left,top=max(0,int(rx1*width)-2),max(0,int(ry1*height)-2)
            right,bottom=min(width,math.ceil(rx2*width)+2),min(height,math.ceil(ry2*height)+2)
            area=(right-left)*(bottom-top)
            if area<9 or area>=changed.size*.8:
                continue
            local_count=int(np.count_nonzero(changed[top:bottom,left:right]))
            # Assess the same rounded measurements that durable replay reads.
            local=round(local_count/area,6)
            background=round((int(np.count_nonzero(changed))-local_count)/(changed.size-area),6)
            if not localized_change_supported(local, background):
                continue
            current["metadata"]["activity_witnesses"].append({
                "kind":kind,"validated":True,"witness_version":LOCALIZED_CHANGE_POLICY_VERSION,"camera_stable":True,
                "from_sample_id":previous["id"],"to_sample_id":current["id"],
                "from_epoch":previous["captured_epoch"],"to_epoch":current["captured_epoch"],
                "observation_ids":[observation["id"]],"normalized_displacement":round(displacement,6),
                "local_change_fraction":round(local,6),"background_change_fraction":round(background,6),
            })
    if confirmation_offsets is not None:
        for offset in confirmation_offsets:
            if any(sample.requested_offset==offset or (sample.requested_offset is None and abs(sample.offset-offset)<1e-6) for sample in samples):
                continue
            captured=round(event_epoch+offset,6)
            records.append({"id":stable_id("sample-",[camera_id,captured,"recorded_main","unavailable"]),
                            "camera_id":camera_id,"captured_epoch":captured,"status":"failed","observations":[],
                            "metadata":{"source":"recorded_main","reason":"recorded_frame_unavailable", "confirmation_complete":True}})
    return sorted(records,key=lambda item:(item["captured_epoch"],item["id"]))
