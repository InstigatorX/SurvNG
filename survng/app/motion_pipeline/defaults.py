from __future__ import annotations

from .factory import MotionStageConfig


def adaptive_motion_stage_configs() -> list[MotionStageConfig]:
    return [
        MotionStageConfig(stage_id="preprocess", implementation="gray_blur"),
        MotionStageConfig(
            stage_id="background",
            implementation="adaptive_ema_background",
            options={
                "learning_rate": 0.025,
                "fast_learning_rate": 0.18,
                "motion_learning_scale": 0.03,
                "global_change_ratio": 0.55,
            },
        ),
        MotionStageConfig(
            stage_id="threshold",
            implementation="adaptive_statistical_threshold",
            options={"sigma": 4.0, "minimum": 7.0, "maximum": 72.0, "smoothing": 0.25},
        ),
        MotionStageConfig(
            stage_id="morphology",
            implementation="open_close",
            options={"kernel_size": 3, "close_iterations": 2},
        ),
        MotionStageConfig(
            stage_id="blob_extract",
            implementation="connected_component_blobs",
            options={"edge_margin_ratio": 0.06},
        ),
        MotionStageConfig(
            stage_id="blob_filter",
            implementation="adaptive_blob_filter",
            options={
                "minimum_area_ratio": 0.00025,
                "minimum_fill_ratio": 0.12,
                "maximum_aspect_ratio": 12.0,
                "ignored_zone_overlap": 0.6,
            },
        ),
        MotionStageConfig(
            stage_id="tracking",
            implementation="persistent_centroid_tracker",
            options={
                "maximum_distance": 0.18,
                "maximum_missed_frames": 3,
                "maximum_track_age_seconds": 8.0,
            },
        ),
        MotionStageConfig(
            stage_id="scoring",
            implementation="adaptive_motion_score",
            options={"minimum_persistence_frames": 2},
        ),
    ]


def default_motion_stage_configs() -> list[MotionStageConfig]:
    return [
        MotionStageConfig(stage_id="preprocess", implementation="gray_blur"),
        MotionStageConfig(stage_id="difference", implementation="frame_difference"),
        MotionStageConfig(
            stage_id="threshold",
            implementation="fixed_threshold",
            options={"value": 18},
        ),
        MotionStageConfig(
            stage_id="morphology",
            implementation="open_close",
            options={"kernel_size": 3, "close_iterations": 2},
        ),
        MotionStageConfig(stage_id="blob_extract", implementation="contour_blobs"),
        MotionStageConfig(
            stage_id="blob_filter",
            implementation="minimum_area",
            options={"minimum_area_ratio": 0.0003},
        ),
        MotionStageConfig(
            stage_id="tracking",
            implementation="dominant_centroid",
            options={
                "minimum_active_area_ratio": 0.0008,
                "minimum_changed_ratio": 0.003,
            },
        ),
        MotionStageConfig(stage_id="scoring", implementation="default_motion_score"),
    ]


def default_motion_observation_stage_configs(
    *,
    mog2_enabled: bool,
    sample_fps: float,
    mog2_history_seconds: float,
) -> list[MotionStageConfig]:
    return [
        MotionStageConfig(
            stage_id="mog2_source",
            implementation="opencv_mog2_evidence",
            options={
                "enabled": mog2_enabled,
                "sample_fps": sample_fps,
                "history_seconds": mog2_history_seconds,
            },
            parallel_group="evidence_sources",
        ),
        MotionStageConfig(
            stage_id="onvif_source",
            implementation="onvif_event_evidence",
            options={
                "enabled": True,
                "base_score": 0.55,
                "priority_score": 0.95,
                "priority_keywords": [
                    "manual",
                    "person",
                    "people",
                    "human",
                    "vehicle",
                    "animal",
                    "face",
                ],
            },
            parallel_group="evidence_sources",
        ),
    ]


def default_motion_fusion_stage_configs() -> list[MotionStageConfig]:
    return [
        MotionStageConfig(
            stage_id="evidence_fusion",
            implementation="buffered_evidence_fusion",
            options={"sources": ["mog2", "onvif"], "policy": "audit"},
        ),
        MotionStageConfig(
            stage_id="event_state",
            implementation="score_event_state",
            options={
                "activation_frames": 1,
                "release_frames": 1,
                "cooldown_seconds": 0.0,
                "state_timeout_seconds": 10.0,
            },
        ),
        MotionStageConfig(stage_id="trigger", implementation="score_trigger"),
    ]
