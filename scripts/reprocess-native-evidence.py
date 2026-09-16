#!/usr/bin/env python3
"""Rebuild native covers in an explicit time window, verifying shortlisted images."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survng.app.config import load_config
from survng.app.event_store import EventStore
from survng.app.image_storage import DurableImageWriter
from survng.app.media_storage import MediaStorageRegistry
from survng.app.native_evidence import NativeEvidenceService


class RecordingIndex:
    def __init__(self, path):
        self.path = path
    def recording_at(self, camera_id, epoch, *, source):
        with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("select * from recordings where camera_id=? and source=? and playable=1 and start_epoch<=? and end_epoch>=? order by start_epoch desc limit 1", (camera_id, source, epoch, epoch)).fetchone()
            return dict(row) if row else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--hours", type=float, default=2)
    parser.add_argument("--report", required=True)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    if not 0 < args.hours <= 24 or not 1 <= args.limit <= 10000:
        parser.error("hours must be within (0,24] and limit within [1,10000]")
    config = load_config(args.config)
    config_root = Path(args.config).resolve().parent
    config.detector.model_path = str(config_root / config.detector.resolved_model_path())
    if config.detector.labels_path:
        config.detector.labels_path = str(config_root / config.detector.labels_path)
    storage = Path(config.storage_dir)
    database = Path(config.database_dir) if config.database_dir else storage
    media = MediaStorageRegistry(storage, config.media_storage)
    events = EventStore(storage, database_dir=database, media_storage=media)
    service = NativeEvidenceService(config, events, RecordingIndex((Path(config.recording_index_dir) if config.recording_index_dir else storage)/"recordings.sqlite3"), DurableImageWriter(config.image_storage), media)
    end = datetime.now(timezone.utc)
    start = end-timedelta(hours=args.hours)
    with sqlite3.connect(f"file:{events.db_path}?mode=ro", uri=True) as conn:
        ids = [row[0] for row in conn.execute("select id from events where created_at>=? and created_at<? and topic='native/object-presence' order by id desc limit ?", (start.isoformat(),end.isoformat(),args.limit))]
    report = {"start":start.isoformat(),"end":end.isoformat(),"selected":len(ids),"results":[]}
    def process(event_id):
        try:
            return service.process(event_id)
        except Exception as error:
            return {"event_id":event_id,"status":"failed","error":str(error)[:300]}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(process, ids):
            report["results"].append(result)
            report["counts"] = dict(Counter(x["status"] for x in report["results"]))
            Path(args.report).write_text(json.dumps(report,indent=2))
            print(json.dumps(result),flush=True)
    Path(args.report).write_text(json.dumps(report,indent=2))
    service.stop()
    print(json.dumps(report.get("counts",{})),flush=True)

if __name__ == "__main__":
    main()
