from __future__ import annotations

import unittest

from pydantic import ValidationError

from survng.app.config import AppConfig, CameraMotionQualificationConfig, MotionQualificationConfig
from survng.app.manager import validate_motion_pipeline_configuration
from survng.app.motion_pipeline import (
    MotionPipelineFactory,
    build_builtin_motion_registry,
    resolve_motion_pipeline_graphs,
)


class MotionPipelineConfigurationTest(unittest.TestCase):
    def test_empty_global_graphs_resolve_to_compatible_defaults(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig(mode="enforce"),
            CameraMotionQualificationConfig(),
        )

        self.assertEqual(graphs.origins, {
            "qualification": "default",
            "observation": "default",
            "fusion": "default",
        })
        self.assertEqual(graphs.qualification[0].implementation, "gray_blur")
        self.assertEqual(graphs.observation[0].implementation, "opencv_mog2_evidence")
        self.assertFalse(graphs.observation[0].options["enabled"])
        self.assertEqual(graphs.observation[1].implementation, "onvif_event_evidence")
        self.assertEqual(graphs.fusion[0].implementation, "buffered_evidence_fusion")
        self.assertEqual(graphs.fusion[0].options["sources"], ["mog2", "onvif"])

    def test_global_graph_is_selected_and_camera_graph_takes_precedence(self) -> None:
        global_config = MotionQualificationConfig.model_validate({
            "pipeline": {
                "qualification": [{
                    "stage_id": "global_qualification",
                    "implementation": "legacy_qualifier",
                    "options": {"profile": "global"},
                }],
            },
        })
        inherited = resolve_motion_pipeline_graphs(
            global_config,
            CameraMotionQualificationConfig(),
        )
        overridden = resolve_motion_pipeline_graphs(
            global_config,
            CameraMotionQualificationConfig.model_validate({
                "pipeline": {
                    "qualification": [{
                        "stage_id": "camera_qualification",
                        "implementation": "legacy_qualifier",
                        "options": {"profile": "camera"},
                    }],
                },
            }),
        )

        self.assertEqual(inherited.origins["qualification"], "global")
        self.assertEqual(inherited.qualification[0].stage_id, "global_qualification")
        self.assertEqual(overridden.origins["qualification"], "camera")
        self.assertEqual(overridden.qualification[0].stage_id, "camera_qualification")
        self.assertEqual(overridden.qualification[0].options["profile"], "camera")

    def test_camera_graph_cannot_explicitly_override_with_empty_list(self) -> None:
        camera_config = CameraMotionQualificationConfig.model_validate({
            "pipeline": {"qualification": []},
        })

        with self.assertRaisesRegex(ValueError, "qualification graph cannot be empty"):
            resolve_motion_pipeline_graphs(MotionQualificationConfig(), camera_config)

    def test_duplicate_stage_ids_are_rejected_by_configuration_schema(self) -> None:
        with self.assertRaises(ValidationError):
            AppConfig.model_validate({
                "motion_qualification": {
                    "pipeline": {
                        "qualification": [
                            {"stage_id": "same", "implementation": "gray_blur"},
                            {"stage_id": "same", "implementation": "frame_difference"},
                        ],
                    },
                },
            })

    def test_unknown_and_incompatible_configured_stages_fail_factory_validation(self) -> None:
        factory = MotionPipelineFactory(build_builtin_motion_registry())
        unknown = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "pipeline": {
                    "qualification": [{
                        "stage_id": "custom",
                        "implementation": "not_registered",
                    }],
                },
            }),
            CameraMotionQualificationConfig(),
        )
        incompatible = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "pipeline": {
                    "qualification": [{
                        "stage_id": "score_first",
                        "implementation": "default_motion_score",
                    }],
                },
            }),
            CameraMotionQualificationConfig(),
        )

        with self.assertRaisesRegex(ValueError, "unknown motion stage implementation"):
            factory.create("gate", unknown.qualification)
        with self.assertRaisesRegex(ValueError, "dominant_track"):
            factory.create("gate", incompatible.qualification)

    def test_resolved_implementation_and_options_are_exposed_in_status(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "pipeline": {
                    "qualification": [{
                        "stage_id": "qualification",
                        "implementation": "legacy_qualifier",
                        "options": {"diagnostic": True},
                    }],
                },
            }),
            CameraMotionQualificationConfig(),
        )
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            graphs.qualification,
        )

        self.assertEqual(pipeline.status()["configuration"], [{
            "stage_id": "qualification",
            "implementation": "legacy_qualifier",
            "options": {"diagnostic": True},
        }])

    def test_application_preflight_rejects_invalid_graph_before_runtime(self) -> None:
        config = AppConfig.model_validate({
            "motion_qualification": {
                "pipeline": {
                    "qualification": [{
                        "stage_id": "custom",
                        "implementation": "not_registered",
                    }],
                },
            },
            "cameras": [{
                "id": "gate",
                "name": "Gate",
                "stream_url": "rtsp://example.invalid/main",
            }],
        })

        with self.assertRaisesRegex(
            ValueError,
            "invalid motion pipeline for camera 'global'",
        ):
            validate_motion_pipeline_configuration(config)

    def test_application_preflight_accepts_registered_camera_override(self) -> None:
        config = AppConfig.model_validate({
            "cameras": [{
                "id": "gate",
                "name": "Gate",
                "stream_url": "rtsp://example.invalid/main",
                "motion_qualification": {
                    "pipeline": {
                        "qualification": [{
                            "stage_id": "qualification",
                            "implementation": "legacy_qualifier",
                        }],
                    },
                },
            }],
        })

        validate_motion_pipeline_configuration(config)

    def test_application_preflight_requires_final_trigger_decision(self) -> None:
        config = AppConfig.model_validate({
            "motion_qualification": {
                "pipeline": {
                    "fusion": [{
                        "stage_id": "fusion_only",
                        "implementation": "buffered_evidence_fusion",
                    }],
                },
            },
        })

        with self.assertRaisesRegex(ValueError, "required artifacts: decision"):
            validate_motion_pipeline_configuration(config)


if __name__ == "__main__":
    unittest.main()
