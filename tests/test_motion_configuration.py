from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from survng.app.config import AppConfig, CameraMotionQualificationConfig, MotionQualificationConfig
from survng.app.manager import validate_manager_configuration, validate_motion_pipeline_configuration
from survng.app.motion_pipeline import (
    MotionPipelineFactory,
    adaptive_motion_stage_configs,
    build_builtin_motion_registry,
    guided_fusion_settings,
    identify_analysis_preset,
    resolve_motion_pipeline_graphs,
    resolved_trigger_mode,
)


class MotionPipelineConfigurationTest(unittest.TestCase):
    def test_application_preflight_requires_complete_media_role_coverage(self) -> None:
        config = AppConfig.model_validate({
            "media_storage": {
                "locations": [{
                    "id": "audit-only",
                    "path": "/mnt/audit-only",
                    "roles": ["motion_audits"],
                }],
            },
        })

        with self.assertRaisesRegex(
            ValueError,
            "enabled media location must accept: recordings, snapshots, clips, exports",
        ):
            validate_manager_configuration(config)

    def test_application_preflight_accepts_roles_split_across_locations(self) -> None:
        config = AppConfig.model_validate({
            "media_storage": {
                "locations": [{
                    "id": "recordings",
                    "path": "/mnt/recordings",
                    "roles": ["recordings", "clips"],
                }, {
                    "id": "evidence",
                    "path": "/mnt/evidence",
                    "roles": ["snapshots", "motion_audits", "exports"],
                }],
            },
        })

        validate_manager_configuration(config)

    def test_guided_helpers_identify_the_adaptive_preset(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig(),
            CameraMotionQualificationConfig(),
        )
        self.assertEqual(identify_analysis_preset(graphs.qualification), "adaptive")
        self.assertEqual(guided_fusion_settings(graphs.fusion)["policy"], "audit")

    def test_fusion_policies_without_sources_are_rejected(self) -> None:
        # With no evidence source, any/all/weighted found nothing to combine
        # and fail-open accepted every motion with score 1.0.
        for policy in ("any", "all", "weighted"):
            config = MotionQualificationConfig.model_validate({
                "mode": "camera",
                "pipeline": {
                    "fusion": [{
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": policy, "sources": []},
                    }],
                },
            })
            with self.subTest(policy=policy), self.assertRaisesRegex(
                ValueError, "motion fusion policy .* is unsupported",
            ):
                resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())

    def test_fusion_rejects_unsupported_sources(self) -> None:
        for sources in (" AUX ", ["depth_object"]):
            config = MotionQualificationConfig.model_validate({
                "mode": "adaptive",
                "pipeline": {
                    "fusion": [{
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "audit", "sources": sources},
                    }],
                },
            })
            with self.subTest(sources=sources), self.assertRaisesRegex(
                ValueError, "motion fusion sources .* are unsupported",
            ):
                resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())

    def test_saved_ui_fusion_options_still_load(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "mode": "camera",
                "pipeline": {
                    "fusion": [{
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {
                            "policy": " BYPASS ",
                            "sources": [],
                            "include_primary": False,
                            "fail_open": True,
                            "source_thresholds": {"onvif": 0.5},
                            "source_weights": {"primary": 1, "onvif": 1},
                            "weighted_threshold": 0.5,
                        },
                    }],
                },
            }),
            CameraMotionQualificationConfig(),
        )
        self.assertEqual(guided_fusion_settings(graphs.fusion)["policy"], "bypass")
        validate_motion_pipeline_configuration(AppConfig.model_validate({
            "motion_qualification": {
                "mode": "camera",
                "pipeline": {"fusion": [{
                    "stage_id": "evidence_fusion",
                    "implementation": "buffered_evidence_fusion",
                    "options": {
                        "policy": "bypass",
                        "include_primary": False,
                        "source_weights": {"primary": 1, "onvif": 1},
                    },
                }]},
            },
        }))

    def test_retired_fixed_threshold_stages_are_rejected(self) -> None:
        config = MotionQualificationConfig.model_validate({
            "pipeline": {
                "qualification": [
                    {"stage_id": "preprocess", "implementation": "gray_blur"},
                    {"stage_id": "difference", "implementation": "frame_difference"},
                ],
            },
        })
        with self.assertRaisesRegex(ValueError, "retired fixed-threshold"):
            resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())

    def test_empty_global_graphs_resolve_to_compatible_defaults(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig(mode="adaptive"),
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
        self.assertEqual(len(graphs.fusion), 1)
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


    def test_scalar_onvif_source_cannot_bypass_new_mode_separation(self) -> None:
        config = MotionQualificationConfig.model_validate({
            "mode": "camera",
            "pipeline": {
                "fusion": [
                    {
                        "stage_id": "evidence_fusion",
                        "implementation": "buffered_evidence_fusion",
                        "options": {"policy": "all", "sources": " ONVIF "},
                    }],
            },
        })

        with self.assertRaisesRegex(ValueError, "ONVIF cannot be a validation source"):
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
                    }],
            },
        })

        with self.assertRaisesRegex(ValueError, "ONVIF cannot be a validation source"):
            resolve_motion_pipeline_graphs(config, CameraMotionQualificationConfig())


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
                            {"stage_id": "same", "implementation": "frame_difference"}],
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
                        "implementation": "adaptive_motion_score",
                    }],
                },
            }),
            CameraMotionQualificationConfig(),
        )

        with self.assertRaisesRegex(ValueError, "unknown motion stage implementation"):
            factory.create("gate", unknown.qualification)
        with self.assertRaisesRegex(ValueError, "tracked_objects"):
            factory.create("gate", incompatible.qualification)

    def test_resolved_implementation_and_options_are_exposed_in_status(self) -> None:
        graphs = resolve_motion_pipeline_graphs(
            MotionQualificationConfig.model_validate({
                "pipeline": {
                    "qualification": [{
                        "stage_id": "qualification",
                        "implementation": "gray_blur",
                        "options": {"blur_kernel_size": 7},
                    }],
                },
            }),
            CameraMotionQualificationConfig(),
        )
        pipeline = MotionPipelineFactory(build_builtin_motion_registry()).create(
            "gate",
            graphs.qualification,
        )

        self.assertEqual(
            pipeline.status()["configuration"][0]["implementation"],
            "gray_blur",
        )
        self.assertEqual(graphs.origins["qualification"], "global")

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
                        "qualification": [
                            {
                                "stage_id": stage.stage_id,
                                "implementation": stage.implementation,
                                "options": dict(stage.options),
                            }
                            for stage in adaptive_motion_stage_configs()
                        ],
                    },
                },
            }],
        })

        validate_motion_pipeline_configuration(config)

    def test_application_preflight_rejects_retired_mog2_flag(self) -> None:
        with self.assertRaisesRegex(
            ValidationError,
            "mog2_audit_enabled",
        ):
            AppConfig.model_validate({
                "motion_qualification": {"mog2_audit_enabled": True},
            })


    def test_application_preflight_rejects_unsupported_fusion_source(self) -> None:
        config = AppConfig.model_validate({
            "cameras": [{
                "id": "gate",
                "name": "Gate",
                "stream_url": "rtsp://example.invalid/main",
                "motion_qualification": {
                    "pipeline": {
                        "fusion": [{
                            "stage_id": "fusion",
                            "implementation": "buffered_evidence_fusion",
                            "options": {"sources": " MOG2 "},
                        }],
                    },
                },
            }],
        })

        with self.assertRaisesRegex(ValueError, r"camera .gate.*fusion sources \[.mog2.\] are unsupported"):
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

    def test_application_preflight_accepts_evidence_only_fusion(self) -> None:
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
                        }],
                },
            },
        })

        with self.assertRaisesRegex(
            ValueError,
            "stage 'threshold'.*option 'sigma'.*valid number",
        ):
            validate_motion_pipeline_configuration(config)


if __name__ == "__main__":
    unittest.main()
