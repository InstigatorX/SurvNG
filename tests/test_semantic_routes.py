from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import ValidationError

from survng.app.semantic_routes import (
    SemanticRouteDependencies,
    SemanticVisualSearchRequest,
    SemanticVisualFrameSearchRequest,
    create_semantic_router,
    normalized_crop_bounds,
)


class SemanticVisualFrameRouteTests(TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.preview_path = Path(self.temporary.name) / "preview.jpg"
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :100] = (0, 0, 255)
        image[:, 100:] = (0, 255, 0)
        self.assertTrue(cv2.imwrite(str(self.preview_path), image))

        self.search_image = Mock(return_value=[
            SimpleNamespace(
                event_id=9,
                score=0.82,
                rank_score=0.81,
                match_strength="strong_match",
                component_scores={"full": 0.82},
                source_kind="object_crop",
                source_key="person:0",
                object_label="person",
                bbox=[1, 2, 3, 4],
            )
        ])
        self.search_event_object = Mock(return_value=[])
        self.preview = Mock(return_value=self.preview_path)
        self.manager = SimpleNamespace(
            camera=lambda camera_id: object() if camera_id == "gate" else None,
            recorder=SimpleNamespace(
                recording_rows_between=Mock(return_value=[{
                    "start_epoch": 100.0,
                    "end_epoch": 110.0,
                    "path": "gate.mp4",
                }])
            ),
            semantic_search=SimpleNamespace(
                search_image=self.search_image,
                search_event_object=self.search_event_object,
            ),
            events=SimpleNamespace(
                get=Mock(return_value={
                    "id": 8,
                    "camera_id": "gate",
                    "kind": "object",
                    "created_at": "2026-08-27T15:59:00+00:00",
                    "objects_json": '[{"label":"person","bbox":[1,2,3,4]}]',
                }),
                get_many=Mock(return_value=[{
                    "id": 9,
                    "camera_id": "yard",
                    "kind": "object",
                    "created_at": "2026-08-27T16:00:00+00:00",
                }])
            ),
            faces=None,
            config=SimpleNamespace(
                base_path="/survng",
                semantic_search=SimpleNamespace(max_results=50),
            ),
        )
        dependencies = SemanticRouteDependencies(
            get_manager=lambda: self.manager,
            manager_lock=threading.RLock(),
            recording_preview_path=self.preview,
        )
        self.handler = create_semantic_router(dependencies).handlers[
            "semantic_visual_frame_search"
        ]
        self.event_handler = create_semantic_router(dependencies).handlers[
            "semantic_visual_search"
        ]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_crop_math_clamps_epsilon_at_image_edge(self) -> None:
        self.assertEqual(
            normalized_crop_bounds(
                200,
                100,
                x=0.75,
                y=0.5,
                width=0.2500005,
                height=0.5000005,
            ),
            (150, 50, 200, 100),
        )

    def test_visual_frame_request_rejects_crop_outside_preview(self) -> None:
        with self.assertRaises(ValidationError):
            SemanticVisualFrameSearchRequest.model_validate({
                "camera_id": "gate",
                "epoch": 105.0,
                "x": 0.8,
                "y": 0.1,
                "width": 0.3,
                "height": 0.5,
            })
        self.assertEqual(self.preview.call_count, 0)

    def test_visual_frame_search_crops_exact_preview_and_hydrates_hits(self) -> None:
        payload = self.handler(SemanticVisualFrameSearchRequest.model_validate({
            "camera_id": "gate",
            "epoch": 105.25,
            "source": "main",
            "x": 0.5,
            "y": 0.25,
            "width": 0.25,
            "height": 0.5,
            "camera_ids": ["yard"],
            "source_kinds": ["object_crop"],
            "exclude_event_id": 8,
        }))

        self.assertEqual(payload["query_mode"], "visual")
        self.assertEqual(payload["crop"], {
            "x": 0.5,
            "y": 0.25,
            "width": 0.25,
            "height": 0.5,
        })
        self.assertEqual(payload["results"][0]["event"]["id"], 9)
        self.assertEqual(
            payload["results"][0]["snapshot_url"],
            "/survng/api/events/9/snapshot.jpg",
        )
        preview_args, preview_kwargs = self.preview.call_args
        self.assertIs(preview_args[0], self.manager)
        self.assertEqual(preview_args[2], 105.25)
        self.assertEqual(preview_kwargs, {"width": 1280, "exact": True})
        crop = self.search_image.call_args.args[0]
        self.assertEqual(crop.shape, (50, 50, 3))
        self.assertTrue(np.all(crop[:, :, 1] > 200))
        self.assertEqual(
            self.search_image.call_args.kwargs["source_kinds"],
            ["object_crop"],
        )
        self.assertEqual(self.search_image.call_args.kwargs["limit"], 50)
        self.assertEqual(
            self.search_image.call_args.kwargs["exclude_event_ids"],
            [8],
        )
        self.assertTrue(self.search_image.call_args.kwargs["unique_events"])

    def test_visual_event_search_excludes_anchor_before_unique_result_limit(self) -> None:
        self.search_event_object.return_value = [
            SimpleNamespace(
                event_id=8,
                score=0.99,
                rank_score=0.99,
                match_strength="visual_similarity",
                component_scores={"full": 0.99},
                source_kind="object_crop",
                source_key="person:1",
                object_label="person",
                bbox=[2, 3, 4, 5],
            ),
            self.search_image.return_value[0],
            SimpleNamespace(
                event_id=9,
                score=0.80,
                rank_score=0.80,
                match_strength="visual_similarity",
                component_scores={"full": 0.80},
                source_kind="full_frame",
                source_key="frame",
                object_label="",
                bbox=None,
            ),
        ]

        payload = self.event_handler(SemanticVisualSearchRequest.model_validate({
            "event_id": 8,
            "object_index": 0,
            "limit": 1,
        }))

        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["results"][0]["event"]["id"], 9)
        call = self.search_event_object.call_args
        self.assertEqual(call.kwargs["limit"], 1)
        self.assertEqual(call.kwargs["exclude_event_ids"], [8])
        self.assertTrue(call.kwargs["unique_events"])

    def test_preview_http_error_is_preserved(self) -> None:
        self.preview.side_effect = HTTPException(
            status_code=429,
            detail="preview capacity is busy",
        )

        with self.assertRaises(HTTPException) as raised:
            self.handler(SemanticVisualFrameSearchRequest.model_validate({
                "camera_id": "gate",
                "epoch": 105.0,
                "x": 0,
                "y": 0,
                "width": 1,
                "height": 1,
            }))

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.detail, "preview capacity is busy")
