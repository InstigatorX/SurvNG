import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import HTTPException

from survng.app.events import EventStore
from survng.app.image_cache import LocalImageCache
from survng.app.incident_queries import IncidentQueryDependencies, IncidentQueryService, create_incident_query_router


def endpoints(tmp_path):
    store=EventStore(tmp_path)
    manager=SimpleNamespace(events=store,storage_dir=tmp_path,media_storage=None,image_cache=LocalImageCache(tmp_path/'cache'))
    bundle=create_incident_query_router(IncidentQueryDependencies(lambda:manager,threading.RLock()),IncidentQueryService())
    return store,{route.path:route.endpoint for route in bundle.router.routes}


def test_observation_windows_and_missing_detail(tmp_path):
    _,routes=endpoints(tmp_path)
    listing=routes['/api/observations']
    for args in ({'status':'excluded'},{'start_at':'2026-01-01'},{'start_at':'2026-01-01T00:00:00Z','end_at':'2026-03-01T00:00:00Z'}):
        with pytest.raises(HTTPException) as error:
            listing(**args)
        assert error.value.status_code==422
    assert listing()['items']==[]
    with pytest.raises(HTTPException) as error:
        routes['/api/observations/{record_id}']('missing')
    assert error.value.status_code==404


def test_unadmitted_acquisition_image_is_available_at_original_and_thumbnail_sizes(tmp_path):
    store,routes=endpoints(tmp_path)
    path=tmp_path/'original.webp'
    cv2.imwrite(str(path),np.full((512,896,3),80,dtype=np.uint8))
    sample=store.acquire_scene_sample(sample_id='original',camera_id='gate',captured_epoch=1000,source='live_discovery',status='complete',snapshot_path=str(path),
        observations=[{'label':'person','confidence':.27,'snapshot_path':str(path)}])
    observation_id=sample['observation_ids'][0]
    snapshot=routes['/api/incidents/observations/{observation_id}/snapshot']
    original=snapshot(observation_id)
    thumbnail=snapshot(observation_id,width=320,quality=82)
    assert cv2.imread(str(original.path)).shape[:2]==(512,896)
    assert cv2.imread(str(thumbnail.path)).shape[1]==320
    assert routes['/api/observations/{sample_id}/snapshot']('original').path==path
    with store._connect() as conn:
        store._expire_acquired_snapshots(conn,[sample['snapshot_path']])
    for route,identifier in [(snapshot,observation_id),(routes['/api/observations/{sample_id}/snapshot'],'original')]:
        with pytest.raises(HTTPException) as error:
            route(identifier)
        assert error.value.status_code==404
