from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from survng.app.config import AppConfig, CameraMotionQualificationConfig, MotionQualificationConfig
from survng.app.manager import validate_motion_pipeline_configuration
from survng.app.motion_pipeline import (
    MotionPipelineFactory,
    analysis_preset_selections,
    build_builtin_motion_registry,
    guided_fusion_settings,
    identify_analysis_preset,
    resolve_motion_pipeline_graphs,
    resolved_trigger_mode,
    update_guided_fusion,
)


class MotionPipelineConfigurationTest(unittest.TestCase):
    def test_guided_helpers_identify_presets_and_update_fusion(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig(),
            CameraMotionQualificationConfig(),
        )
        self.assertEqual(identify_analysis_preset(graphs.qualification), "adaptive")
        self.assertEqual(
            analysis_preset_selections("classic")[0].implementation,
            "legacy_qualifier",
        )

        updated = update_guided_fusion(graphs.fusion, "fusion_policy", "weighted")
        resolved = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "pipeline": {"fusion": [stage.model_dump() for stage in updated]},
            }),
            CameraMotionQualificationConfig(),
        )
        self.assertEqual(guided_fusion_settings(resolved.fusion)["policy"], "weighted")

    def test_guided_fusion_normalizes_scalar_source_for_status(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "mode": "enforce",
                "pipeline": {
                    "fusion": [
                        {
                            "stage_id": "evidence_fusion",
                            "implementation": "buffered_evidence_fusion",
                            "options": {"policy": " ALL ", "sources": " MOG2 "},
                        },
                        {"stage_id": "event_state", "implementation": "score_event_state"},
                        {"stage_id": "trigger", "implementation": "score_trigger"},
                    ],
                },
            }),
            CameraMotionQualificationConfig(),
        )

        settings = guided_fusion_settings(graphs.fusion)
        self.assertEqual(settings["policy"], "all")
        self.assertEqual(settings["sources"], ["mog2"])

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
        self.assertEqual(graphs.qualification[1].implementation, "adaptive_ema_background")
        self.assertEqual(len(graphs.observation), 1)
        self.assertEqual(graphs.observation[0].implementation, "onvif_event_evidence")
        self.assertEqual(graphs.fusion[0].implementation, "buffered_evidence_fusion")
        self.assertEqual(graphs.fusion[0].options["sources"], [])
        self.assertTrue(graphs.fusion[0].options["include_primary"])
        self.assertTrue(graphs.fusion[0].options["fail_open"])

    def test_camera_visual_backup_mode_is_preserved_through_resolution(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig(mode="camera_rescue"),
            CameraMotionQualificationConfig(),
        )

        self.assertEqual(resolved_trigger_mode("camera_rescue"), "camera_rescue")
        self.assertTrue(graphs.fusion[0].options["include_primary"])

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

    def test_mog2_observation_runs_when_enforced_fusion_uses_it(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "mode": "enforce",
                "mog2_audit_enabled": True,
                "pipeline": {
                    "fusion": [
                        {
                            "stage_id": "evidence_fusion",
                            "implementation": "buffered_evidence_fusion",
                            "options": {"policy": "any", "sources": ["mog2", "onvif"]},
                        },
                        {
                            "stage_id": "event_state",
                            "implementation": "score_event_state",
                        },
                        {"stage_id": "trigger", "implementation": "score_trigger"},
                    ],
                },
            }),
            CameraMotionQualificationConfig(),
        )

        self.assertEqual(graphs.observation[0].implementation, "opencv_mog2_evidence")
        self.assertTrue(graphs.observation[0].options["enabled"])

    def test_mog2_decision_graph_is_authoritative_over_legacy_enable_flag(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "mode": "camera",
                "mog2_audit_enabled": False,
                "pipeline": {
                    "fusion": [
                        {
                            "stage_id": "evidence_fusion",
                            "implementation": "buffered_evidence_fusion",
                            "options": {"policy": "all", "sources": ["mog2"]},
                        },
                        {"stage_id": "event_state", "implementation": "score_event_state"},
                        {"stage_id": "trigger", "implementation": "score_trigger"},
                    ],
                },
            }),
            CameraMotionQualificationConfig(),
        )

        self.assertEqual(graphs.observation[0].implementation, "opencv_mog2_evidence")
        self.assertTrue(graphs.observation[0].options["enabled"])

    def test_mog2_audit_graph_is_authoritative_over_legacy_enable_flag(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "mode": "camera",
                "mog2_audit_enabled": False,
                "pipeline": {
                    "fusion": [
                        {
                            "stage_id": "evidence_fusion",
                            "implementation": "buffered_evidence_fusion",
                            "options": {"policy": "audit", "sources": ["mog2"]},
                        },
                        {"stage_id": "event_state", "implementation": "score_event_state"},
                        {"stage_id": "trigger", "implementation": "score_trigger"},
                    ],
                },
            }),
            CameraMotionQualificationConfig(),
        )

        self.assertEqual(graphs.observation[0].implementation, "opencv_mog2_evidence")

    def test_scalar_mog2_source_is_normalized_during_graph_resolution(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "mode": "camera",
                "pipeline": {
                    "fusion": [
                        {
                            "stage_id": "evidence_fusion",
                            "implementation": "buffered_evidence_fusion",
                            "options": {"policy": "all", "sources": " MOG2 "},
                        },
                        {"stage_id": "event_state", "implementation": "score_event_state"},
                        {"stage_id": "trigger", "implementation": "score_trigger"},
                    ],
                },
            }),
            CameraMotionQualificationConfig(),
        )

        self.assertEqual(graphs.observation[0].implementation, "opencv_mog2_evidence")

    def test_scalar_onvif_source_cannot_bypass_new_mode_separation(self) -> None:
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "all", "sources": " ONVIF "},
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })

        with self.assertRaisesRegex(ValueError, "ONVIF cannot be a validation source"):
            resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())

    def test_bypass_graph_does_not_run_unused_mog2_source(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "mode": "camera",
                "pipeline": {
                    "fusion": [
                        {
                            "stage_id": "evidence_fusion",
                            "implementation": "buffered_evidence_fusion",
                            "options": {"policy": "bypass", "sources": ["mog2"]},
                        },
                        {"stage_id": "event_state", "implementation": "score_event_state"},
                        {"stage_id": "trigger", "implementation": "score_trigger"},
                    ],
                },
            }),
            CameraMotionQualificationConfig(),
        )

        self.assertEqual(len(graphs.observation), 1)
        self.assertEqual(graphs.observation[0].implementation, "onvif_event_evidence")

    def test_custom_observation_graph_cannot_omit_required_mog2_source(self) -> None:
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "pipeline": {
                "observation": [{
                    "stage_id": "onvif_source",
                    "implementation": "onvif_event_evidence",
                }],
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "all", "sources": ["mog2"]},
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })

        with self.assertRaisesRegex(ValueError, "no enabled MOG2 source"):
            resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())

    def test_custom_observation_graph_cannot_disable_required_mog2_source(self) -> None:
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "pipeline": {
                "observation": [{
                    "stage_id": "mog2_source",
                    "implementation": "opencv_mog2_evidence",
                    "options": {"enabled": False},
                }],
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "all", "sources": ["mog2"]},
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })

        with self.assertRaisesRegex(ValueError, "no enabled MOG2 source"):
            resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())

    def test_new_modes_reject_onvif_as_a_validation_source(self) -> None:
        config = MotionQualificationConfig.model_validate({
            "mode": "adaptive",
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "all", "sources": ["onvif"]},
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })

        with self.assertRaisesRegex(ValueError, "ONVIF cannot be a validation source"):
            resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())

    def test_legacy_mode_keeps_onvif_fusion_loadable_for_migration(self) -> None:
        config = MotionQualificationConfig.model_validate({
            "mode": "enforce",
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "all", "sources": ["onvif"]},
                    },
                    {"stage_id": "event_state", "implementation": "score_event_state"},
                    {"stage_id": "trigger", "implementation": "score_trigger"},
                ],
            },
        })

        graphs = resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())
        self.assertEqual(graphs.fusion[0].options["sources"], ["onvif"])

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

    def test_application_preflight_closes_temporary_pipelines(self) -> None:
        config = AppConfig.model_validate({
            "cameras": [{
                "id": "gate",
                "name": "Gate",
                "stream_url": "rtsp://example.invalid/main",
            }],
        })
        pipeline = Mock()

        with patch(
            "survng.app.manager.MotionPipelineFactory.create",
            return_value=pipeline,
        ) as create:
            validate_motion_pipeline_configuration(config)

        self.assertEqual(create.call_count, 6)
        self.assertEqual(pipeline.close.call_count, 6)

    def test_application_preflight_closes_prior_pipeline_after_failure(self) -> None:
        config = AppConfig()
        pipeline = Mock()

        with (
            patch(
                "survng.app.manager.MotionPipelineFactory.create",
                side_effect=[pipeline, ValueError("invalid test graph")],
            ),
            self.assertRaisesRegex(ValueError, "invalid test graph"),
        ):
            validate_motion_pipeline_configuration(config)

        pipeline.close.assert_called_once_with()

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

    def test_application_preflight_reports_invalid_stage_options_with_context(self) -> None:
        config = AppConfig.model_validate({
            "motion_qualification": {
                "pipeline": {
                    "qualification": [
                        {"stage_id": "preprocess", "implementation": "gray_blur"},
                        {
                            "stage_id": "background",
                            "implementation": "adaptive_ema_background",
                        },
                        {
                            "stage_id": "threshold",
                            "implementation": "adaptive_statistical_threshold",
                            "options": {"sigma": None},
                        },
                    ],
                },
            },
        })

        with self.assertRaisesRegex(
            ValueError,
            "stage 'threshold'.*option 'sigma'.*valid number",
        ):
            validate_motion_pipeline_configuration(config)

    def test_application_preflight_rejects_nonfinite_nested_fusion_values(self) -> None:
        config = AppConfig.model_validate({
            "motion_qualification": {
                "pipeline": {
                    "fusion": [{
                        "stage_id": "fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {
                            "source_thresholds": {"mog2": float("nan")},
                        },
                    }, {
                        "stage_id": "state",
                        "implementation": "score_event_state",
                    }, {
                        "stage_id": "trigger",
                        "implementation": "score_trigger",
                    }],
                },
            },
        })

        with self.assertRaisesRegex(
            ValueError,
            "source threshold for mog2 must be finite",
        ):
            validate_motion_pipeline_configuration(config)


if __name__ == "__main__":
    unittest.main()
