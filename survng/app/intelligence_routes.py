from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import threading
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .assistant import AssistantAnswer, AssistantChatRequest, AssistantEvidence, AssistantProvider, AssistantToolCall, IncidentVisualReviewer
from .assistant_investigation import correlate_incident_timeline
from .audit_ai import AuditAiAdvisor, AuditAiChange, AuditAiError, ai_provider_configured, motion_audit_interpretation, motion_paradigm_context, validate_tuning_value
from .calibration import apply_calibration_changes, build_calibration_report, calibration_configuration_fingerprint, calibration_setting_value
from .camera_intelligence import aggregate_camera_intelligence, compare_camera_intelligence_results, select_balanced_samples
from .config import AppConfig, CameraConfig, camera_by_id
from .incident_presenter import _event_row, _incident_rows
from .incident_queries import IncidentQueryService, _filter_incident_summaries, _motion_audit_row
from .incident_utils import DEFAULT_INCIDENT_GAP_SECONDS, event_snapshot_path, snapshot_media_type
from .manager import AppManager, validate_motion_pipeline_configuration
from .motion_ai_review import aggregate_motion_ai_review
from .motion_pipeline import guided_fusion_settings, identify_analysis_preset, resolve_motion_pipeline_graphs
from .recording_routes import recording_source
from .security import redact_secret_text

LOGGER = logging.getLogger(__name__)
AI_RECOMMENDATION_SECRET = secrets.token_bytes(32)
AI_RECOMMENDATION_MAX_AGE_SECONDS = 60 * 60
CALIBRATION_MODE_LIMITS = {
    "quick": (24.0, 100, 12),
    "standard": (168.0, 100, 20),
    "deep": (720.0, 100, 40),
}

class AuditAiApplyRequest(BaseModel):
    changes: list[AuditAiChange] = Field(default_factory=list, max_length=8)
    confirmed: bool = False
    configuration_fingerprint: str = Field(default='', max_length=64)
    recommendation_proof: str = Field(default='', max_length=256)

class IncidentAiApplyRequest(BaseModel):
    changes: list[AuditAiChange] = Field(default_factory=list, max_length=8)
    confirmed: bool = False
    configuration_fingerprint: str = Field(default='', max_length=64)
    recommendation_proof: str = Field(default='', max_length=256)

class MotionAiReviewRequest(BaseModel):
    camera_id: str = Field(min_length=1, max_length=128)
    hours: float = Field(default=24.0, ge=1.0, le=168.0)
    record_limit: int = Field(default=100, ge=20, le=100)
    image_limit: int = Field(default=12, ge=4, le=24)

class CameraIntelligenceApplyRequest(IncidentAiApplyRequest):
    evaluation_hours: float = Field(default=24.0, ge=24.0, le=168.0)

class CameraIntelligenceFollowupRequest(BaseModel):
    image_limit: int = Field(default=12, ge=4, le=24)

class CalibrationRunRequest(BaseModel):
    camera_ids: list[str] = Field(default_factory=list, max_length=128)
    mode: str = Field(default='standard', pattern='^(quick|standard|deep)$')
    override_active_evaluation: bool = False

class CalibrationApplyRequest(BaseModel):
    recommendation_ids: list[str] = Field(min_length=1, max_length=256)
    confirmed: bool = False
    configuration_fingerprint: str = Field(min_length=64, max_length=64)
    evaluation_hours: float = Field(default=24.0, ge=24.0, le=168.0)

class CalibrationRollbackRequest(BaseModel):
    change_ids: list[str] = Field(default_factory=list, max_length=256)
    camera_ids: list[str] = Field(default_factory=list, max_length=128)
    confirmed: bool = False
    force_conflicts: bool = False

@dataclass(frozen=True, slots=True)
class IntelligenceDependencies:
    get_config: Callable[[], AppConfig]
    get_manager: Callable[[], AppManager]
    manager_lock: threading.RLock
    get_audit_ai_limiter: Callable[[], Any]
    get_assistant_limiter: Callable[[], Any]
    application_stopping: threading.Event
    incident_queries: IncidentQueryService
    system_telemetry: Any
    apply_config_update: Callable[..., Any]
    begin_ai_operation: Callable[[str], None]
    end_ai_operation: Callable[[str], None]
    media_export_manager: Callable[[], Any]


@dataclass(frozen=True, slots=True)
class IntelligenceRouteBundle:
    router: APIRouter
    service: "IntelligenceService"
    handlers: dict[str, Callable[..., Any]]


class IntelligenceService:

    def __init__(self, deps: IntelligenceDependencies) -> None:
        self.deps = deps

    def _run_registered_ai_worker(
        self,
        operation: str,
        target: Callable[..., None],
        args: tuple[Any, ...],
    ) -> None:
        try:
            target(*args)
        finally:
            self.deps.end_ai_operation(operation)

    def _start_registered_ai_thread(
        self,
        operation: str,
        target: Callable[..., None],
        args: tuple[Any, ...],
        *,
        name: str,
    ) -> threading.Thread:
        """Register manager-bound work before its thread can race a reload."""
        self.deps.begin_ai_operation(operation)
        thread = threading.Thread(
            target=self._run_registered_ai_worker,
            args=(operation, target, args),
            name=name,
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            self.deps.end_ai_operation(operation)
            raise
        return thread

    def _audit_ai_context(self, audit: dict, active_config: AppConfig, active_manager: AppManager) -> dict:
        camera_id = str(audit.get('camera_id') or '')
        camera = camera_by_id(active_config, camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail='audit camera not found')
        try:
            features = json.loads(str(audit.get('features_json') or '{}'))
        except (json.JSONDecodeError, TypeError):
            features = {}
        if not isinstance(features, dict):
            features = {}
        pipeline_telemetry = features.pop('pipeline_telemetry', {})
        event = active_manager.events.get(int(audit['event_id'])) if audit.get('event_id') else None
        detected_objects: list[dict] = []
        qualification: dict = {}
        object_tracking: dict[str, Any] = {}
        if event:
            try:
                entries = json.loads(str(event.get('objects_json') or '[]'))
            except json.JSONDecodeError:
                entries = []
            for entry in entries if isinstance(entries, list) else []:
                if not isinstance(entry, dict):
                    continue
                if entry.get('label'):
                    detected_objects.append({
                        key: entry.get(key)
                        for key in (
                            'label', 'confidence', 'box', 'zones',
                            'incident_eligible', 'incident_admission_reason',
                            'semantic_tier', 'semantic_candidate_threshold',
                            'semantic_rescue_threshold', 'semantic_standard_threshold',
                            'semantic_median_confidence', 'semantic_rescue_admitted',
                            'temporal_consensus', 'temporal_sample_offset_seconds',
                            'temporal_observations', 'temporal_track_observations',
                            'temporal_incident_observations',
                            'temporal_required_observations', 'temporal_samples',
                            'temporal_peak_confidence',
                            'temporal_peak_confidence_offset_seconds',
                            'temporal_label_votes',
                            'temporal_center_displacement_ratio',
                            'temporal_center_path_ratio',
                            'temporal_first_observation_offset_seconds',
                            'temporal_last_observation_offset_seconds',
                            'temporal_newly_appeared',
                            'temporal_robust_new_appearance', 'temporal_zone_entry',
                            'activity_role', 'activity_confidence',
                            'motion_correlated', 'motion_correlation',
                            'motion_correlation_threshold',
                            'motion_temporal_evidence_available',
                            'track_id', 'track_state', 'track_observations',
                        )
                    })
                if entry.get('status') == 'motion_qualification':
                    candidate = entry.get('motion_qualification')
                    qualification = candidate if isinstance(candidate, dict) else {}
                if entry.get('status') == 'object_tracking':
                    candidate = entry.get('object_tracking')
                    object_tracking = candidate if isinstance(candidate, dict) else {}
        override = camera.motion_qualification
        graphs = resolve_motion_pipeline_graphs(active_config.motion_qualification, override)
        effective_mode = active_config.motion_qualification.mode if override.mode == 'inherit' else override.mode
        fusion = guided_fusion_settings(graphs.fusion)
        require_incident_zone = active_config.detector.require_incident_zone if camera.require_incident_zone is None else camera.require_incident_zone
        object_activity_attribution = active_config.detector.object_activity_attribution if camera.object_activity_attribution == 'inherit' else camera.object_activity_attribution
        suppression_verification_rate = active_config.motion_qualification.suppression_verification_rate if override.suppression_verification_rate is None else override.suppression_verification_rate
        visual_backup = {'grace_seconds': active_config.motion_qualification.visual_backup_grace_seconds if override.visual_backup_grace_seconds is None else override.visual_backup_grace_seconds, 'minimum_score': active_config.motion_qualification.visual_backup_min_score if override.visual_backup_min_score is None else override.visual_backup_min_score, 'minimum_consecutive': active_config.motion_qualification.visual_backup_min_consecutive if override.visual_backup_min_consecutive is None else override.visual_backup_min_consecutive, 'cooldown_seconds': active_config.motion_qualification.visual_backup_cooldown_seconds if override.visual_backup_cooldown_seconds is None else override.visual_backup_cooldown_seconds, 'maximum_triggers_5m': active_config.motion_qualification.visual_backup_max_triggers_5m if override.visual_backup_max_triggers_5m is None else override.visual_backup_max_triggers_5m}
        effective = {'mode': effective_mode, 'sensitivity': active_config.motion_qualification.sensitivity if override.sensitivity == 'inherit' else override.sensitivity, 'stationary_object_tolerance': active_config.motion_qualification.stationary_object_tolerance if override.stationary_object_tolerance == 'inherit' else override.stationary_object_tolerance, 'object_activity_attribution': object_activity_attribution, 'illumination_filter_enabled': active_config.motion_qualification.illumination_filter_enabled if override.illumination_filter_enabled is None else override.illumination_filter_enabled, 'frame_width': override.frame_width or active_config.motion_qualification.frame_width, 'borderline_rescue_enabled': active_config.motion_qualification.borderline_rescue_enabled if override.borderline_rescue_enabled is None else override.borderline_rescue_enabled, 'borderline_margin': active_config.motion_qualification.borderline_margin if override.borderline_margin is None else override.borderline_margin, 'sample_fps': active_config.motion_qualification.sample_fps, 'window_seconds': active_config.motion_qualification.window_seconds, 'post_trigger_seconds': active_config.motion_qualification.post_trigger_seconds, 'burst_quiet_seconds': active_config.motion_qualification.burst_quiet_seconds, 'camera_mode_background_fps': active_config.motion_qualification.camera_mode_background_fps, 'visual_backup_warmup_seconds': active_config.motion_qualification.visual_backup_warmup_seconds, 'visual_backup_grace_seconds': visual_backup['grace_seconds'], 'visual_backup_min_score': visual_backup['minimum_score'], 'visual_backup_score_margin': active_config.motion_qualification.visual_backup_score_margin, 'visual_backup_min_consecutive': visual_backup['minimum_consecutive'], 'visual_backup_cooldown_seconds': visual_backup['cooldown_seconds'], 'visual_backup_max_triggers_5m': visual_backup['maximum_triggers_5m'], 'rejected_sample_rate': active_config.motion_qualification.rejected_sample_rate, 'suppression_verification_rate': suppression_verification_rate, 'analysis_preset': identify_analysis_preset(graphs.qualification), 'object_confirmation_frames': active_config.detector.event_confirmation_frames, 'object_class_confirmation_frames': dict(active_config.detector.event_class_confirmation_frames), 'object_class_confidence_thresholds': dict(active_config.detector.event_class_confidence_thresholds), 'incident_eligibility_policy': 'zones_only' if require_incident_zone else 'zones_plus_full_frame', 'object_tracking': {'enabled': active_config.detector.tracking.enabled, 'implementation': active_config.detector.tracking.implementation, 'sample_fps': active_config.detector.tracking.sample_fps, 'reid_enabled': active_config.detector.tracking.reid_enabled, 'vehicle_reid_enabled': active_config.detector.tracking.vehicle_reid_enabled}, 'fusion': fusion, 'pipeline_origins': graphs.origins}
        effective.pop('analysis_preset', None)
        recent, _ = active_manager.events.motion_audits(limit=50, camera_id=camera_id)
        reason_counts: dict[str, int] = {}
        object_matches = 0
        for row in recent:
            reason = str(row.get('reason') or 'unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            object_matches += int(row.get('object_detected') == 1)
        interpretation = motion_audit_interpretation(reason=audit.get('reason'), event_id=audit.get('event_id'), object_detected=audit.get('object_detected'))
        related_prior_event: dict[str, Any] | None = None
        if interpretation['category'] in {'duplicate_active_event', 'duplicate_event_cooldown'}:
            related_id = audit.get('related_event_id')
            prior = active_manager.events.get(int(related_id)) if related_id else None
            try:
                audit_at = datetime.fromisoformat(str(audit.get('created_at')))
            except (TypeError, ValueError):
                audit_at = None
            if prior is None and audit_at is not None:
                nearby_events = active_manager.events.for_camera_range(camera_id, (audit_at - timedelta(seconds=60)).isoformat(), audit_at.isoformat(), limit=50)
                prior = nearby_events[-1] if nearby_events else None
            if prior is not None:
                try:
                    prior_entries = json.loads(str(prior.get('objects_json') or '[]'))
                except (json.JSONDecodeError, TypeError):
                    prior_entries = []
                prior_objects = [{'label': entry.get('label'), 'confidence': entry.get('confidence'), 'incident_eligible': entry.get('incident_eligible', True)} for entry in prior_entries if isinstance(entry, dict) and entry.get('label')] if isinstance(prior_entries, list) else []
                try:
                    if audit_at is None:
                        raise ValueError('audit timestamp unavailable')
                    seconds_before = max(0.0, (audit_at - datetime.fromisoformat(str(prior.get('created_at')))).total_seconds())
                except (TypeError, ValueError):
                    seconds_before = None
                related_prior_event = {'event_id': prior.get('id'), 'created_at': prior.get('created_at'), 'seconds_before': seconds_before, 'objects': prior_objects}
        return {'motion_paradigm': motion_paradigm_context(mode=effective_mode, onvif_enabled=camera.onvif.enabled, has_live_substream=bool(camera.live_stream_url), fusion=fusion, require_incident_zone=require_incident_zone), 'audit': {'id': audit.get('id'), 'camera_id': camera_id, 'created_at': audit.get('created_at'), 'score': audit.get('score'), 'threshold': audit.get('threshold'), 'reason': audit.get('reason'), 'category': audit.get('category') or 'qualification', 'mode': audit.get('mode'), 'sensitivity': audit.get('sensitivity'), 'decision_id': audit.get('decision_id'), 'event_id': audit.get('event_id'), 'related_event_id': audit.get('related_event_id'), 'features': features, 'trigger_count': audit.get('trigger_count'), 'object_detected': None if audit.get('object_detected') is None else bool(audit.get('object_detected')), 'qualification': qualification, 'pipeline_telemetry': pipeline_telemetry}, 'decision_outcome': {'interpretation': interpretation, 'filtered_before_object_detection': bool(not audit.get('event_id') and audit.get('object_detected') is None), 'object_detection_ran': bool(audit.get('event_id') or audit.get('object_detected') is not None), 'object_detection_completed': audit.get('object_detected') is not None, 'object_detected': None if audit.get('object_detected') is None else bool(audit.get('object_detected')), 'eligible_object_found': audit.get('object_detected') == 1, 'visual_backup': audit.get('category') == 'visual_backup'}, 'related_prior_event': related_prior_event, 'detected_objects': detected_objects, 'object_tracking': object_tracking, 'effective_settings': effective, 'recent_camera_audits': {'sample_size': len(recent), 'object_matches': object_matches, 'reason_counts': reason_counts}, 'setting_bounds': {'frame_width': [240, 960], 'sample_fps': [2, 10], 'window_seconds': [0.8, 4], 'post_trigger_seconds': [0.5, 6], 'burst_quiet_seconds': [0.1, 2], 'borderline_margin': [0, 0.1], 'visual_backup_warmup_seconds': [0, 120], 'visual_backup_grace_seconds': [0, 5], 'visual_backup_min_score': [0, 1], 'visual_backup_score_margin': [0, 0.5], 'visual_backup_min_consecutive': [2, 10], 'visual_backup_cooldown_seconds': [5, 300], 'visual_backup_max_triggers_5m': [1, 30], 'sensitivity': ['high', 'balanced', 'low'], 'stationary_object_tolerance': ['low', 'balanced', 'high'], 'analysis_preset': ['adaptive', 'modular']}}

    def _apply_pipeline_ai_change(self, next_config: AppConfig, camera: CameraConfig, change: AuditAiChange, value: object) -> None:
        target = next_config.motion_qualification if change.scope == 'global' else camera.motion_qualification
        setattr(target, change.setting, value)

    def motion_audit(self, limit: int=24, offset: int=0, camera_id: str='', outcome: str='all', category: str='all') -> dict:
        if outcome not in {'all', 'object', 'clear', 'not_run'}:
            raise HTTPException(status_code=400, detail='invalid motion audit outcome')
        if category not in {'all', 'qualification', 'visual_backup', 'active_followup'}:
            raise HTTPException(status_code=400, detail='invalid motion audit category')
        with self.deps.manager_lock:
            active_manager = self.deps.get_manager()
            rows, total = active_manager.events.motion_audits(limit=limit, offset=offset, camera_id=camera_id, outcome=outcome, category=category)
            storage_dir = active_manager.storage_dir
        return {'items': [_motion_audit_row(row, storage_dir) for row in rows], 'total': total, 'limit': max(1, min(int(limit), 100)), 'offset': max(0, int(offset))}

    def motion_effectiveness(self, days: float=7.0) -> dict:
        with self.deps.manager_lock:
            active_events = self.deps.get_manager().events
        return active_events.motion_effectiveness(days=days)

    def motion_audit_snapshot(self, audit_id: int) -> FileResponse:
        with self.deps.manager_lock:
            active_manager = self.deps.get_manager()
            audit = active_manager.events.get_motion_audit(audit_id)
        if audit is None:
            raise HTTPException(status_code=404, detail='motion audit entry not found')
        try:
            snapshot_path = event_snapshot_path(active_manager.storage_dir, audit)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        media_type = snapshot_media_type(snapshot_path)
        return FileResponse(snapshot_path, media_type=media_type, headers={'Cache-Control': 'private, max-age=300'})

    async def motion_audit_ai_analyze(self, audit_id: int) -> dict:
        with self.deps.manager_lock:
            active_manager = self.deps.get_manager()
            active_config = self.deps.get_config().model_copy(deep=True)
            audit_config = active_config.audit_ai
            audit = active_manager.events.get_motion_audit(audit_id)
            if audit is None:
                raise HTTPException(status_code=404, detail='motion audit entry not found')
            if not self.deps.get_audit_ai_limiter().acquire(blocking=False):
                raise HTTPException(status_code=429, detail='an AI audit request is already running', headers={'Retry-After': '5'})
            self.deps.begin_ai_operation('motion_audit')
        try:
            snapshot_path = event_snapshot_path(active_manager.storage_dir, audit)
            analysis_context = self._audit_ai_context(audit, active_config, active_manager)
            advice = await asyncio.to_thread(AuditAiAdvisor(audit_config).analyze, snapshot_path, analysis_context)
            camera = camera_by_id(active_config, str(audit.get('camera_id') or ''))
            if camera is None:
                raise AuditAiError('audit camera is unavailable')
            changes, _previews = self._assistant_motion_change_previews(active_config, camera, [change for change in advice.changes if change.scope == 'camera'])
            advice.changes = changes
            configuration_fingerprint = self._assistant_motion_config_fingerprint(active_config, camera)
            recommendation_proof = self._issue_ai_recommendation_token(kind='motion_audit', record_id=audit_id, camera_id=camera.id, configuration_fingerprint=configuration_fingerprint, changes=changes)
        except AuditAiError as exc:
            raise HTTPException(status_code=502, detail=redact_secret_text(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        finally:
            self.deps.end_ai_operation('motion_audit')
            self.deps.get_audit_ai_limiter().release()
        return {'audit_id': audit_id, 'camera_id': audit.get('camera_id'), 'provider': audit_config.provider, 'model': audit_config.model or '', 'motion_paradigm': analysis_context['motion_paradigm'], 'advice': advice.model_dump(mode='json'), 'configuration_fingerprint': configuration_fingerprint, 'recommendation_proof': recommendation_proof, 'apply_requires_confirmation': True}

    def motion_audit_ai_apply(self, audit_id: int, request: AuditAiApplyRequest) -> dict:
        if not request.confirmed:
            raise HTTPException(status_code=400, detail='explicit confirmation is required')
        if not request.changes:
            raise HTTPException(status_code=400, detail='no recommendation changes supplied')
        if any((change.scope != 'camera' for change in request.changes)):
            raise HTTPException(status_code=400, detail='single motion reviews may only change settings for the reviewed camera')
        with self.deps.manager_lock:
            if not self.deps.get_config().audit_ai.allow_apply_recommendations:
                raise HTTPException(status_code=403, detail='applying AI recommendations is disabled')
            audit = self.deps.get_manager().events.get_motion_audit(audit_id)
            if audit is None:
                raise HTTPException(status_code=404, detail='motion audit entry not found')
            next_config = self.deps.get_config().model_copy(deep=True)
            camera = camera_by_id(next_config, str(audit.get('camera_id') or ''))
            if camera is None:
                raise HTTPException(status_code=404, detail='audit camera not found')
            current_fingerprint = self._assistant_motion_config_fingerprint(next_config, camera)
            if request.configuration_fingerprint != current_fingerprint:
                raise HTTPException(status_code=409, detail='motion settings changed after this review; run AI analysis again')
            if not self._verify_ai_recommendation_token(request.recommendation_proof, kind='motion_audit', record_id=audit_id, camera_id=camera.id, configuration_fingerprint=current_fingerprint, changes=request.changes):
                raise HTTPException(status_code=409, detail='AI recommendations are expired or do not match this review')
            applied: list[dict] = []
            try:
                for change in request.changes:
                    value = validate_tuning_value(change.setting, change.value)
                    self._apply_pipeline_ai_change(next_config, camera, change, value)
                    applied.append({**change.model_dump(mode='json'), 'value': value})
                validate_motion_pipeline_configuration(next_config)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            next_config = AppConfig.model_validate(next_config.model_dump(mode='json'))
            _effective_config, apply_result = self.deps.apply_config_update(next_config)
        return {'ok': True, 'audit_id': audit_id, 'camera_id': camera.id, 'applied': applied, 'workers_restarted': bool(apply_result['camera_workers_restarted']), 'apply_mode': apply_result['apply_mode']}

    def _run_motion_ai_review(self, review_id: int, audits: list[dict[str, Any]], active_config: AppConfig, active_manager: AppManager) -> None:
        analyses: list[tuple[dict[str, Any], Any]] = []
        images_available = 0
        failures = 0
        consecutive_failures = 0
        first_error = ''
        current_settings: dict[str, Any] = {}
        review_context: dict[str, Any] = {}
        try:
            active_manager.events.update_motion_ai_review(review_id, status='running')
            advisor = AuditAiAdvisor(active_config.audit_ai)
            candidates: list[tuple[dict[str, Any], Path]] = []
            for audit in audits:
                try:
                    snapshot_path = event_snapshot_path(active_manager.storage_dir, audit)
                except (FileNotFoundError, PermissionError):
                    continue
                candidates.append((audit, snapshot_path))
            images_available = len(candidates)
            active_manager.events.update_motion_ai_review(review_id, status='running', images_available=images_available, analyzed=0, failed=0)
            for audit, snapshot_path in candidates:
                try:
                    context = self._audit_ai_context(audit, active_config, active_manager)
                    if not current_settings:
                        current_settings = dict(context.get('effective_settings') or {})
                        review_context = {'motion_paradigm': context.get('motion_paradigm') or {}, 'effective_settings': current_settings}
                    context['camera_review'] = {'purpose': 'aggregate camera tuning review', 'instruction': 'Recommend only camera-scoped changes supported by this image.'}
                    advice = advisor.analyze(snapshot_path, context)
                except (AuditAiError, FileNotFoundError, PermissionError) as exc:
                    failures += 1
                    consecutive_failures += 1
                    if not first_error:
                        first_error = redact_secret_text(exc)
                    active_manager.events.update_motion_ai_review(review_id, status='running', images_available=images_available, analyzed=len(analyses), failed=failures)
                    if consecutive_failures >= 3:
                        break
                    continue
                analyses.append((audit, advice))
                consecutive_failures = 0
                active_manager.events.update_motion_ai_review(review_id, status='running', images_available=images_available, analyzed=len(analyses), failed=failures)
            if not analyses:
                if images_available == 0:
                    error = 'No retained motion-audit images are available for this camera'
                else:
                    error = first_error or 'AI analysis did not complete for any retained image'
                active_manager.events.update_motion_ai_review(review_id, status='failed', images_available=images_available, analyzed=0, failed=failures, error=error)
                return
            result = aggregate_motion_ai_review(analyses, audits_considered=len(audits), images_available=images_available, failed=failures, current_settings=current_settings, review_context=review_context)
            active_manager.events.update_motion_ai_review(review_id, status='completed', images_available=images_available, analyzed=len(analyses), failed=failures, result=result, error=first_error if failures else '')
        except Exception as exc:
            logging.getLogger(__name__).exception('motion AI review %s failed', review_id)
            try:
                active_manager.events.update_motion_ai_review(review_id, status='failed', images_available=images_available, analyzed=len(analyses), failed=failures, error=redact_secret_text(exc))
            except Exception:
                logging.getLogger(__name__).exception('failed to persist motion AI review %s failure', review_id)
        finally:
            self.deps.get_audit_ai_limiter().release()

    def _camera_intelligence_candidates(self, camera: CameraConfig, active_manager: AppManager, *, hours: float, record_limit: int, image_limit: int) -> tuple[list[dict[str, Any]], int]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff.isoformat()
        audits, _total = active_manager.events.motion_audits(limit=record_limit, camera_id=camera.id)
        recent_audits = [audit for audit in audits if not audit.get('created_at') or str(audit.get('created_at')) >= cutoff_iso]
        end_iso = datetime.now(timezone.utc).isoformat()
        if hasattr(active_manager.events, 'recent_for_camera_range'):
            event_rows = active_manager.events.recent_for_camera_range(camera.id, cutoff_iso, end_iso, limit=max(500, record_limit * 8))
        elif hasattr(active_manager.events, 'for_camera_range'):
            event_rows = list(reversed(active_manager.events.for_camera_range(camera.id, cutoff_iso, end_iso, limit=max(500, record_limit * 8))))
        else:
            event_rows = []
        incident_summaries = _incident_rows([_event_row(row) for row in event_rows], DEFAULT_INCIDENT_GAP_SECONDS)[:record_limit]
        incidents = self.deps.incident_queries.with_faces(active_manager, self.deps.incident_queries.hydrate(active_manager, incident_summaries)) if incident_summaries else []
        candidates: list[dict[str, Any]] = []
        incident_event_ids: set[int] = set()
        for incident in incidents:
            event_id = int(incident.get('representative_event_id') or 0)
            if event_id <= 0:
                continue
            incident_event_ids.update((int(event.get('id') or 0) for event in incident.get('events') or []))
            candidates.append({'kind': 'incident', 'camera_id': camera.id, 'record_id': event_id, 'event_id': event_id, 'created_at': incident.get('start_at'), 'category': 'recognized_incident' if incident.get('has_objects') or incident.get('labels') else 'motion_only_incident'})
        for audit in recent_audits:
            reason = str(audit.get('reason') or '')
            if reason in {'event_state_active', 'event_state_cooldown'}:
                continue
            linked_event_id = int(audit.get('event_id') or 0)
            if linked_event_id and linked_event_id in incident_event_ids:
                continue
            if audit.get('category') == 'visual_backup':
                category = 'visual_backup'
            elif audit.get('category') == 'active_followup':
                category = 'possible_miss' if audit.get('object_detected') != 1 else 'recognized_incident'
            elif audit.get('object_detected') == 0:
                category = 'possible_miss'
            elif not linked_event_id:
                category = 'motion_filtered'
            else:
                category = 'other'
            candidates.append({'kind': 'motion_decision', 'camera_id': camera.id, 'record_id': int(audit.get('id') or 0), 'audit_id': int(audit.get('id') or 0), 'created_at': audit.get('created_at'), 'category': category, 'audit': audit})
        candidates.sort(key=lambda item: str(item.get('created_at') or ''), reverse=True)
        record_pool = select_balanced_samples(candidates, record_limit)
        return (select_balanced_samples(record_pool, image_limit), len(record_pool))

    def _run_camera_intelligence_review(self, review_id: int, camera_id: str, samples: list[dict[str, Any]], records_considered: int, hours: float, active_config: AppConfig, active_manager: AppManager, evaluation_id: int=0, baseline_result: dict[str, Any] | None=None) -> None:
        analyses: list[dict[str, Any]] = []
        failed = 0
        consecutive_failures = 0
        first_error = ''
        try:
            active_manager.events.update_motion_ai_review(review_id, status='running', images_available=len(samples), analyzed=0, failed=0)
            audit_advisor = AuditAiAdvisor(active_config.audit_ai)
            for sample in samples:
                try:
                    if sample['kind'] == 'incident':
                        evidence = self._assistant_visual_incident_evidence(int(sample['event_id']), active_config, active_manager)
                        if evidence is None:
                            raise AuditAiError('incident evidence is unavailable')
                        advice = evidence.client_data['advice']
                        verdict = {'detection_consistent': 'consistent', 'probable_missed_detection': 'likely_miss', 'probable_misclassification': 'likely_misclassification', 'probable_false_positive': 'likely_false_alarm'}.get(str(advice.get('verdict')), 'uncertain')
                        analyses.append({**sample, 'verdict': verdict, 'confidence': advice.get('confidence'), 'summary': advice.get('summary'), 'visible_subjects': advice.get('visible_subjects') or [], 'detector_assessment': advice.get('detector_assessment'), 'tracking_assessment': advice.get('tracking_assessment'), 'changes': advice.get('changes') or [], 'image_url': evidence.image_url})
                        consecutive_failures = 0
                    else:
                        audit = sample['audit']
                        snapshot_path = event_snapshot_path(active_manager.storage_dir, audit)
                        context = self._audit_ai_context(audit, active_config, active_manager)
                        context['camera_intelligence_review'] = {'purpose': 'balanced cross-incident camera review', 'instruction': 'Recommend only bounded camera-scoped changes this image supports; SurvNG will require repeated support across images.'}
                        advice = audit_advisor.analyze(snapshot_path, context)
                        has_subject = bool(advice.visible_subjects)
                        verdict = 'likely_miss' if advice.verdict == 'real_motion' and has_subject and (audit.get('object_detected') != 1) else 'likely_false_alarm' if advice.verdict == 'noise' else 'consistent' if advice.verdict == 'real_motion' else 'uncertain'
                        audit_id = int(audit['id'])
                        analyses.append({**sample, 'verdict': verdict, 'confidence': advice.confidence, 'summary': advice.summary, 'visible_subjects': advice.visible_subjects, 'detector_assessment': 'missed' if verdict == 'likely_miss' else 'unavailable', 'tracking_assessment': 'unavailable', 'changes': [change.model_dump(mode='json') for change in advice.changes if change.scope == 'camera'], 'image_url': f'/api/motion-audit/{audit_id}/snapshot.jpg'})
                        consecutive_failures = 0
                except (AuditAiError, FileNotFoundError, PermissionError, ValueError) as exc:
                    failed += 1
                    consecutive_failures += 1
                    if not first_error:
                        first_error = redact_secret_text(exc)
                active_manager.events.update_motion_ai_review(review_id, status='running', images_available=len(samples), analyzed=len(analyses), failed=failed)
                if consecutive_failures >= 3:
                    first_error = first_error or 'Camera review stopped after repeated analysis failures'
                    break
            if not analyses:
                active_manager.events.update_motion_ai_review(review_id, status='failed', images_available=len(samples), analyzed=0, failed=failed, error=first_error or 'No retained images could be reviewed')
                if evaluation_id:
                    active_manager.events.reset_camera_intelligence_followup(evaluation_id, first_error or 'No retained images could be reviewed')
                return
            result = aggregate_camera_intelligence(analyses, records_considered=records_considered, selected_images=len(samples), failed=failed, hours=hours)
            camera = camera_by_id(active_config, camera_id)
            if camera is not None:
                proposed_changes = [AuditAiChange(scope='camera', setting=item['setting'], value=item['value'], reason=(item.get('reasons') or ['Repeated review evidence supports this change.'])[0]) for item in result.get('recommendations') or []]
                _normalized, previews = self._assistant_motion_change_previews(active_config, camera, proposed_changes)
                preview_by_key = {(item['setting'], json.dumps(item['proposed'], sort_keys=True)): item for item in previews}
                recommendations = []
                for item in result.get('recommendations') or []:
                    preview = preview_by_key.get((item['setting'], json.dumps(item['value'], sort_keys=True)))
                    if preview:
                        recommendations.append({**item, **preview})
                result['recommendations'] = recommendations
                result['configuration_fingerprint'] = self._assistant_motion_config_fingerprint(active_config, camera)
                result['can_apply'] = bool(recommendations and active_config.audit_ai.allow_apply_recommendations)
            if evaluation_id and baseline_result is not None:
                comparison = compare_camera_intelligence_results(baseline_result, result)
                result['effectiveness_comparison'] = comparison
                active_manager.events.complete_camera_intelligence_evaluation(evaluation_id, followup_result=result, comparison=comparison)
            active_manager.events.update_motion_ai_review(review_id, status='completed', images_available=len(samples), analyzed=len(analyses), failed=failed, result=result, error=first_error if failed else '')
        except Exception as exc:
            LOGGER.exception('camera intelligence review %s failed', review_id)
            try:
                active_manager.events.update_motion_ai_review(review_id, status='failed', analyzed=len(analyses), failed=failed, error=redact_secret_text(exc))
            except Exception:
                LOGGER.exception('failed to persist camera intelligence review %s failure', review_id)
            if evaluation_id:
                try:
                    active_manager.events.reset_camera_intelligence_followup(evaluation_id, redact_secret_text(exc))
                except Exception:
                    LOGGER.exception('failed to reset camera intelligence evaluation %s', evaluation_id)
        finally:
            self.deps.end_ai_operation('camera_intelligence')
            self.deps.get_audit_ai_limiter().release()

    def start_motion_ai_review(self, request: MotionAiReviewRequest) -> dict:
        with self.deps.manager_lock:
            active_manager = self.deps.get_manager()
            active_config = self.deps.get_config().model_copy(deep=True)
            camera = camera_by_id(active_config, request.camera_id)
            if camera is None:
                raise HTTPException(status_code=404, detail='camera not found')
            if not active_config.audit_ai.enabled:
                raise HTTPException(status_code=400, detail='AI audit advisor is disabled')
            if not ai_provider_configured(active_config.audit_ai):
                raise HTTPException(status_code=400, detail='AI audit API key is not configured')
            samples, records_considered = self._camera_intelligence_candidates(camera, active_manager, hours=request.hours, record_limit=request.record_limit, image_limit=request.image_limit)
            if not samples:
                raise HTTPException(status_code=404, detail='this camera has no recent incidents or motion decisions to review')
            if not self.deps.get_audit_ai_limiter().acquire(blocking=False):
                raise HTTPException(status_code=429, detail='an AI audit or camera review is already running', headers={'Retry-After': '5'})
            self.deps.begin_ai_operation('camera_intelligence')
            try:
                review = active_manager.events.create_motion_ai_review(camera.id, records_considered)
            except BaseException:
                self.deps.end_ai_operation('camera_intelligence')
                self.deps.get_audit_ai_limiter().release()
                raise
        thread = threading.Thread(target=self._run_camera_intelligence_review, args=(int(review['id']), camera.id, samples, records_considered, request.hours, active_config, active_manager), name=f'camera-intelligence-{camera.id}', daemon=True)
        try:
            thread.start()
        except BaseException:
            self.deps.end_ai_operation('camera_intelligence')
            self.deps.get_audit_ai_limiter().release()
            active_manager.events.update_motion_ai_review(int(review['id']), status='failed', error='AI review worker could not start')
            raise
        return review

    def latest_motion_ai_review(self, camera_id: str) -> dict:
        with self.deps.manager_lock:
            active_config = self.deps.get_config()
            active_events = self.deps.get_manager().events
            camera = camera_by_id(active_config, camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail='camera not found')
        review = active_events.latest_motion_ai_review(camera.id)
        return review or {'camera_id': camera.id, 'status': 'never'}

    def motion_ai_review(self, review_id: int) -> dict:
        with self.deps.manager_lock:
            review = self.deps.get_manager().events.get_motion_ai_review(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail='motion AI review not found')
        return review

    def camera_intelligence_apply(self, review_id: int, request: CameraIntelligenceApplyRequest) -> dict:
        if not request.confirmed:
            raise HTTPException(status_code=400, detail='explicit confirmation is required')
        if not request.changes:
            raise HTTPException(status_code=400, detail='no recommendation changes supplied')
        if any((change.scope != 'camera' for change in request.changes)):
            raise HTTPException(status_code=400, detail='camera intelligence may only change settings for the reviewed camera')
        with self.deps.manager_lock:
            if not self.deps.get_config().audit_ai.allow_apply_recommendations:
                raise HTTPException(status_code=403, detail='applying AI recommendations is disabled')
            review = self.deps.get_manager().events.get_motion_ai_review(review_id)
            if review is None:
                raise HTTPException(status_code=404, detail='camera intelligence review not found')
            result = review.get('result') or {}
            if review.get('status') != 'completed' or result.get('review_type') != 'camera_intelligence':
                raise HTTPException(status_code=409, detail='camera intelligence review is not complete')
            next_config = self.deps.get_config().model_copy(deep=True)
            camera = camera_by_id(next_config, str(review.get('camera_id') or ''))
            if camera is None:
                raise HTTPException(status_code=404, detail='reviewed camera not found')
            current_fingerprint = self._assistant_motion_config_fingerprint(next_config, camera)
            if request.configuration_fingerprint != current_fingerprint:
                raise HTTPException(status_code=409, detail='motion settings changed after this review; run camera intelligence again')
            persisted: dict[tuple[str, str], dict[str, Any]] = {}
            for recommendation in result.get('recommendations') or []:
                setting = str(recommendation.get('setting') or '')
                value = recommendation.get('proposed', recommendation.get('value'))
                persisted[setting, json.dumps(value, sort_keys=True)] = recommendation
            approved_changes: list[AuditAiChange] = []
            for submitted in request.changes:
                try:
                    normalized_value = validate_tuning_value(submitted.setting, submitted.value)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                recommendation = persisted.get((submitted.setting, json.dumps(normalized_value, sort_keys=True)))
                if recommendation is None:
                    raise HTTPException(status_code=400, detail=f'{submitted.setting} is not an unchanged recommendation from this review')
                approved_changes.append(AuditAiChange(scope='camera', setting=submitted.setting, value=normalized_value, reason=str((recommendation.get('reasons') or [recommendation.get('reason') or 'Repeated evidence supports this change.'])[0])))
            try:
                changes, previews = self._assistant_motion_change_previews(next_config, camera, approved_changes)
                if not changes:
                    raise ValueError('recommendations do not change active settings')
                for change in changes:
                    self._apply_pipeline_ai_change(next_config, camera, change, change.value)
                validate_motion_pipeline_configuration(next_config)
                next_config = AppConfig.model_validate(next_config.model_dump(mode='json'))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            active_events = self.deps.get_manager().events
            _effective_config, apply_result = self.deps.apply_config_update(next_config)
            evaluation = active_events.create_camera_intelligence_evaluation(camera_id=camera.id, baseline_review_id=review_id, evaluation_hours=request.evaluation_hours, applied_changes=previews, baseline_result=result)
        return {'ok': True, 'review_id': review_id, 'camera_id': camera.id, 'applied': previews, 'workers_restarted': bool(apply_result['camera_workers_restarted']), 'apply_mode': apply_result['apply_mode'], 'effectiveness_evaluation': evaluation}

    def latest_camera_intelligence_evaluation(self, camera_id: str) -> dict:
        with self.deps.manager_lock:
            camera = camera_by_id(self.deps.get_config(), camera_id)
            active_events = self.deps.get_manager().events
        if camera is None:
            raise HTTPException(status_code=404, detail='camera not found')
        evaluation = active_events.latest_camera_intelligence_evaluation(camera.id)
        return evaluation or {'camera_id': camera.id, 'status': 'never'}

    def start_camera_intelligence_followup(self, evaluation_id: int, request: CameraIntelligenceFollowupRequest) -> dict:
        with self.deps.manager_lock:
            active_manager = self.deps.get_manager()
            active_config = self.deps.get_config().model_copy(deep=True)
            evaluation = active_manager.events.get_camera_intelligence_evaluation(evaluation_id)
            if evaluation is None:
                raise HTTPException(status_code=404, detail='effectiveness evaluation not found')
            if evaluation.get('status') != 'ready':
                detail = f'follow-up evidence is still being collected until {evaluation.get('ready_at')}' if evaluation.get('status') == 'collecting' else 'effectiveness follow-up is already running or complete'
                raise HTTPException(status_code=409, detail=detail)
            camera = camera_by_id(active_config, str(evaluation.get('camera_id') or ''))
            if camera is None:
                raise HTTPException(status_code=404, detail='reviewed camera not found')
            if not active_config.audit_ai.enabled or not ai_provider_configured(active_config.audit_ai):
                raise HTTPException(status_code=400, detail='AI analysis is not configured')
            try:
                applied_at = datetime.fromisoformat(str(evaluation['applied_at']).replace('Z', '+00:00'))
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=500, detail='evaluation start time is invalid') from exc
            if applied_at.tzinfo is None:
                applied_at = applied_at.replace(tzinfo=timezone.utc)
            elapsed_hours = max(1.0, (datetime.now(timezone.utc) - applied_at).total_seconds() / 3600.0)
            samples, records_considered = self._camera_intelligence_candidates(camera, active_manager, hours=min(168.0, elapsed_hours), record_limit=100, image_limit=request.image_limit)
            if not samples:
                raise HTTPException(status_code=404, detail='no follow-up camera images are available')
            if not self.deps.get_audit_ai_limiter().acquire(blocking=False):
                raise HTTPException(status_code=429, detail='another AI camera review is already running', headers={'Retry-After': '5'})
            self.deps.begin_ai_operation('camera_intelligence')
            try:
                review = active_manager.events.create_motion_ai_review(camera.id, records_considered)
                active_manager.events.start_camera_intelligence_followup(evaluation_id, int(review['id']))
            except BaseException:
                self.deps.end_ai_operation('camera_intelligence')
                self.deps.get_audit_ai_limiter().release()
                raise
        thread = threading.Thread(target=self._run_camera_intelligence_review, args=(int(review['id']), camera.id, samples, records_considered, min(168.0, elapsed_hours), active_config, active_manager, evaluation_id, evaluation.get('baseline_result') or {}), name=f'camera-effectiveness-{camera.id}', daemon=True)
        try:
            thread.start()
        except BaseException:
            self.deps.end_ai_operation('camera_intelligence')
            self.deps.get_audit_ai_limiter().release()
            active_manager.events.update_motion_ai_review(int(review['id']), status='failed', error='Effectiveness review worker could not start')
            active_manager.events.reset_camera_intelligence_followup(evaluation_id, 'Effectiveness review worker could not start')
            raise
        return active_manager.events.get_camera_intelligence_evaluation(evaluation_id) or {}

    def _calibration_camera_review(self, camera: CameraConfig, *, hours: float, record_limit: int, image_limit: int, active_config: AppConfig, active_manager: AppManager) -> dict[str, Any]:
        samples, records_considered = self._camera_intelligence_candidates(camera, active_manager, hours=hours, record_limit=record_limit, image_limit=image_limit)
        if not samples:
            return {'review_type': 'camera_intelligence', 'summary': 'No retained incidents or motion decisions were available.', 'analyzed': 0, 'failed': 0, 'recommendations': [], 'samples': []}
        wait_started = time.monotonic()
        while not self.deps.get_audit_ai_limiter().acquire(timeout=5):
            if self.deps.application_stopping.is_set():
                raise RuntimeError('camera review stopped because SurvNG is shutting down')
            if time.monotonic() - wait_started >= 300:
                raise RuntimeError('timed out waiting for the AI review worker')
        self.deps.begin_ai_operation('camera_intelligence')
        try:
            review = active_manager.events.create_motion_ai_review(camera.id, records_considered)
        except BaseException:
            self.deps.end_ai_operation('camera_intelligence')
            self.deps.get_audit_ai_limiter().release()
            raise
        self._run_camera_intelligence_review(int(review['id']), camera.id, samples, records_considered, hours, active_config, active_manager)
        completed = active_manager.events.get_motion_ai_review(int(review['id'])) or {}
        if completed.get('status') != 'completed':
            raise RuntimeError(str(completed.get('error') or 'camera review failed'))
        return {**(completed.get('result') or {}), 'source_review_id': int(review['id'])}

    def _run_system_calibration(self, run_id: int, camera_ids: list[str], mode: str, active_config: AppConfig, active_manager: AppManager) -> None:
        hours, record_limit, image_limit = CALIBRATION_MODE_LIMITS[mode]
        reports: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        try:
            active_manager.events.update_calibration_run(run_id, status='running')
            for camera_id in camera_ids:
                if self.deps.application_stopping.is_set():
                    raise RuntimeError('system calibration stopped because SurvNG is shutting down')
                camera = camera_by_id(active_config, camera_id)
                if camera is None:
                    errors[camera_id] = 'camera is no longer configured'
                    continue
                try:
                    reports[camera_id] = self._calibration_camera_review(camera, hours=hours, record_limit=record_limit, image_limit=image_limit, active_config=active_config, active_manager=active_manager)
                except Exception as exc:
                    LOGGER.warning('calibration review failed for %s', camera_id, exc_info=True)
                    errors[camera_id] = redact_secret_text(exc)
                active_manager.events.update_calibration_run(run_id, status='running', result={'progress': {'completed': len(reports) + len(errors), 'total': len(camera_ids)}, 'camera_errors': errors})
            if not reports:
                raise RuntimeError('no selected camera could be analyzed')
            statuses = {str(item.get('id') or ''): item for item in active_manager.statuses()}
            stream_health = {camera_id: {key: statuses.get(camera_id, {}).get(key) for key in ('running', 'live_fps', 'main_fps', 'capture_read_failures', 'analysis_frames_dropped', 'last_error') if key in statuses.get(camera_id, {})} for camera_id in reports}
            result = build_calibration_report(active_config, reports, mode=mode, stream_health=stream_health)
            result['camera_errors'] = errors
            result['progress'] = {'completed': len(camera_ids), 'total': len(camera_ids)}
            active_manager.events.update_calibration_run(run_id, status='completed', result=result)
        except Exception as exc:
            LOGGER.exception('system calibration run %s failed', run_id)
            active_manager.events.update_calibration_run(run_id, status='interrupted' if self.deps.application_stopping.is_set() else 'failed', result={'camera_reports': reports, 'camera_errors': errors}, error=redact_secret_text(exc))

    def start_calibration_run(self, request: CalibrationRunRequest) -> dict:
        with self.deps.manager_lock:
            active_manager = self.deps.get_manager()
            active_config = self.deps.get_config().model_copy(deep=True)
            if not active_config.audit_ai.enabled or not ai_provider_configured(active_config.audit_ai):
                raise HTTPException(status_code=400, detail='AI analysis is not configured')
            active_runs = [item for item in active_manager.events.calibration_runs(20) if item.get('status') in {'queued', 'running'}]
            if active_runs:
                raise HTTPException(status_code=409, detail='a system calibration analysis is already running')
            if not request.override_active_evaluation:
                active_sets = [item for item in active_manager.events.calibration_change_sets(100) if item.get('status') in {'collecting', 'reviewing'}]
                if active_sets:
                    raise HTTPException(status_code=409, detail='a calibration change set is still being evaluated; explicitly override it to start another run')
            available = {camera.id for camera in active_config.cameras}
            camera_ids = list(dict.fromkeys(request.camera_ids or [camera.id for camera in active_config.cameras]))
            unknown = sorted(set(camera_ids) - available)
            if unknown:
                raise HTTPException(status_code=404, detail=f'unknown cameras: {', '.join(unknown)}')
            if not camera_ids:
                raise HTTPException(status_code=400, detail='no cameras are configured')
            run = active_manager.events.create_calibration_run(mode=request.mode, camera_ids=camera_ids, configuration_fingerprint=calibration_configuration_fingerprint(active_config))
        try:
            self._start_registered_ai_thread(
                'calibration',
                self._run_system_calibration,
                (int(run['id']), camera_ids, request.mode, active_config, active_manager),
                name=f'survng-calibration-{run['id']}',
            )
        except BaseException as exc:
            active_manager.events.update_calibration_run(int(run['id']), status='failed', error=f'Calibration worker could not start: {redact_secret_text(exc)}')
            raise HTTPException(status_code=503, detail='calibration worker could not start') from exc
        return run

    def calibration_runs(self, limit: int=20) -> dict:
        return {'runs': self.deps.get_manager().events.calibration_runs(limit, include_result=False)}

    def calibration_run(self, run_id: int) -> dict:
        run = self.deps.get_manager().events.get_calibration_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail='calibration run not found')
        return run

    def calibration_apply(self, run_id: int, request: CalibrationApplyRequest) -> dict:
        if not request.confirmed:
            raise HTTPException(status_code=400, detail='explicit confirmation is required')
        with self.deps.manager_lock:
            if not self.deps.get_config().audit_ai.allow_apply_recommendations:
                raise HTTPException(status_code=403, detail='applying AI recommendations is disabled')
            active_events = self.deps.get_manager().events
            run = active_events.get_calibration_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail='calibration run not found')
            if run.get('status') != 'completed':
                raise HTTPException(status_code=409, detail='calibration analysis is not complete')
            current_fingerprint = calibration_configuration_fingerprint(self.deps.get_config())
            if request.configuration_fingerprint != current_fingerprint or run.get('configuration_fingerprint') != current_fingerprint:
                raise HTTPException(status_code=409, detail='calibration settings changed after analysis; run calibration again')
            recommendations = {str(item.get('id') or ''): item for item in (run.get('result') or {}).get('recommendations') or []}
            recommendation_ids = list(dict.fromkeys(request.recommendation_ids))
            unknown = [item for item in recommendation_ids if item not in recommendations]
            if unknown:
                raise HTTPException(status_code=400, detail='one or more recommendations changed or expired')
            selected = [recommendations[item] for item in recommendation_ids]
            try:
                next_config, changes = apply_calibration_changes(self.deps.get_config(), selected)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not changes:
                raise HTTPException(status_code=409, detail='selected recommendations no longer change configuration')
            for index, change in enumerate(changes, start=1):
                change['id'] = f'{run_id}:{index}:{change['setting']}:{change.get('camera_id') or 'global'}'
            before_config = self.deps.get_config().model_copy(deep=True)
            before_fingerprint = current_fingerprint
            _effective, apply_result = self.deps.apply_config_update(next_config)
            after_fingerprint = calibration_configuration_fingerprint(self.deps.get_config())
            try:
                change_set = active_events.create_calibration_change_set(run_id=run_id, parent_change_set_id=None, action='apply', status='collecting', evaluation_hours=request.evaluation_hours, configuration_fingerprint_before=before_fingerprint, configuration_fingerprint_after=after_fingerprint, changes=changes, apply_result=apply_result)
            except BaseException:
                try:
                    self.deps.apply_config_update(before_config)
                except Exception:
                    LOGGER.exception('calibration ledger failure rollback was incomplete')
                raise
        return {'ok': True, 'change_set': change_set}

    def calibration_change_sets(self, limit: int=50) -> dict:
        with self.deps.manager_lock:
            active_events = self.deps.get_manager().events
            rows = active_events.calibration_change_sets(limit)
        now = datetime.now(timezone.utc)
        for row in rows:
            if row.get('action') == 'apply':
                row['rolled_back_change_ids'] = sorted(active_events.calibration_rollback_change_ids(int(row['id'])))
            try:
                created = datetime.fromisoformat(str(row.get('created_at') or '').replace('Z', '+00:00'))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                ready_at = created + timedelta(hours=float(row.get('evaluation_hours') or 24))
                row['ready_at'] = ready_at.isoformat()
                row['seconds_until_ready'] = max(0, round((ready_at - now).total_seconds()))
            except (TypeError, ValueError):
                row['ready_at'] = ''
                row['seconds_until_ready'] = 0
        return {'change_sets': rows}

    def _run_calibration_evaluation(self, change_set_id: int, active_config: AppConfig, active_manager: AppManager) -> None:
        change_set = active_manager.events.get_calibration_change_set(change_set_id) or {}
        run = active_manager.events.get_calibration_run(int(change_set.get('run_id') or 0)) or {}
        baseline_reports = (run.get('result') or {}).get('camera_reports') or {}
        global_change = any((item.get('scope') == 'global' for item in change_set.get('changes') or []))
        affected = {str(item.get('camera_id') or '') for item in change_set.get('changes') or [] if item.get('camera_id')}
        if global_change:
            affected.update((str(item) for item in run.get('camera_ids') or []))
        comparisons: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        try:
            with self.deps.manager_lock:
                current_fingerprint = calibration_configuration_fingerprint(self.deps.get_config())
            expected_fingerprint = str(change_set.get('configuration_fingerprint_after') or '')
            if expected_fingerprint and current_fingerprint != expected_fingerprint:
                active_manager.events.update_calibration_evaluation(change_set_id, {'outcome': 'inconclusive', 'summary': 'Calibration settings changed again before follow-up evidence was reviewed, so this change set cannot be evaluated independently.', 'comparison_basis': 'configuration_conflict'}, status='evaluated')
                return
            try:
                applied_at = datetime.fromisoformat(str(change_set.get('created_at') or '').replace('Z', '+00:00'))
                if applied_at.tzinfo is None:
                    applied_at = applied_at.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                applied_at = datetime.now(timezone.utc) - timedelta(hours=24)
            hours = max(1.0, min(720.0, (datetime.now(timezone.utc) - applied_at).total_seconds() / 3600.0))
            mode = str(run.get('mode') or 'standard')
            _baseline_hours, record_limit, image_limit = CALIBRATION_MODE_LIMITS.get(mode, CALIBRATION_MODE_LIMITS['standard'])
            for camera_id in sorted(affected):
                if self.deps.application_stopping.is_set():
                    raise RuntimeError('calibration follow-up stopped because SurvNG is shutting down')
                camera = camera_by_id(active_config, camera_id)
                baseline = baseline_reports.get(camera_id)
                if camera is None or not isinstance(baseline, dict):
                    errors[camera_id] = 'baseline evidence is unavailable'
                    continue
                try:
                    followup = self._calibration_camera_review(camera, hours=hours, record_limit=record_limit, image_limit=image_limit, active_config=active_config, active_manager=active_manager)
                    comparisons[camera_id] = {'comparison': compare_camera_intelligence_results(baseline, followup), 'followup': followup}
                except Exception as exc:
                    LOGGER.warning('calibration follow-up failed for %s', camera_id, exc_info=True)
                    errors[camera_id] = redact_secret_text(exc)
            outcomes = Counter((str(item.get('comparison', {}).get('outcome') or 'inconclusive') for item in comparisons.values()))
            comparable = outcomes['improved'] + outcomes['worsened']
            if not comparisons or comparable == 0:
                outcome = 'inconclusive'
            elif outcomes['worsened'] and outcomes['improved']:
                outcome = 'mixed'
            elif outcomes['worsened']:
                outcome = 'regressed'
            else:
                outcome = 'improved'
            evaluation = {'outcome': outcome, 'camera_comparisons': comparisons, 'camera_errors': errors, 'summary': f'{outcomes['improved']} cameras improved, {outcomes['worsened']} regressed, and {outcomes['inconclusive']} were inconclusive using matched before/after evidence.', 'comparison_basis': 'category_matched_balanced_samples'}
            with self.deps.manager_lock:
                final_fingerprint = calibration_configuration_fingerprint(self.deps.get_config())
            if expected_fingerprint and final_fingerprint != expected_fingerprint:
                evaluation = {'outcome': 'inconclusive', 'summary': 'Calibration settings changed while follow-up evidence was being reviewed, so this result cannot be attributed to one change set.', 'comparison_basis': 'configuration_conflict', 'camera_errors': errors}
            active_manager.events.update_calibration_evaluation(change_set_id, evaluation, status='evaluated')
        except Exception as exc:
            LOGGER.exception('calibration evaluation %s failed', change_set_id)
            active_manager.events.update_calibration_evaluation(change_set_id, {'outcome': 'failed', 'error': redact_secret_text(exc)}, status='evaluation_failed')

    async def _calibration_followup_monitor(self) -> None:
        """Start due follow-ups automatically without overlapping evaluations."""
        while not self.deps.application_stopping.is_set():
            await asyncio.sleep(60)
            if self.deps.application_stopping.is_set():
                return
            selected: tuple[int, AppConfig, AppManager] | None = None
            with self.deps.manager_lock:
                active_manager = self.deps.get_manager()
                change_sets = active_manager.events.calibration_change_sets(100)
                if any((item.get('status') == 'reviewing' for item in change_sets)):
                    continue
                now = datetime.now(timezone.utc)
                for item in reversed(change_sets):
                    if item.get('action') != 'apply' or item.get('status') != 'collecting':
                        continue
                    try:
                        created = datetime.fromisoformat(str(item.get('created_at') or '').replace('Z', '+00:00'))
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=timezone.utc)
                        ready_at = created + timedelta(hours=float(item.get('evaluation_hours') or 24))
                    except (TypeError, ValueError):
                        continue
                    if now < ready_at:
                        continue
                    change_set_id = int(item['id'])
                    active_manager.events.update_calibration_change_set_status(change_set_id, 'reviewing')
                    selected = (change_set_id, self.deps.get_config().model_copy(deep=True), active_manager)
                    break
            if selected is not None:
                change_set_id, active_config, active_manager = selected
                try:
                    self._start_registered_ai_thread(
                        'calibration_evaluation',
                        self._run_calibration_evaluation,
                        (change_set_id, active_config, active_manager),
                        name=f'survng-calibration-evaluation-{change_set_id}',
                    )
                except BaseException as exc:
                    LOGGER.exception('calibration evaluation worker could not start')
                    active_manager.events.update_calibration_evaluation(change_set_id, {'outcome': 'failed', 'error': f'Evaluation worker could not start: {redact_secret_text(exc)}'}, status='evaluation_failed')

    def start_calibration_evaluation(self, change_set_id: int) -> dict:
        with self.deps.manager_lock:
            active_manager = self.deps.get_manager()
            active_config = self.deps.get_config().model_copy(deep=True)
            change_set = active_manager.events.get_calibration_change_set(change_set_id)
            if change_set is None:
                raise HTTPException(status_code=404, detail='calibration change set not found')
            if change_set.get('action') != 'apply':
                raise HTTPException(status_code=400, detail='rollback entries are not evaluated')
            if change_set.get('status') not in {'collecting', 'evaluation_failed'}:
                raise HTTPException(status_code=409, detail='change set is already evaluating or complete')
            try:
                created = datetime.fromisoformat(str(change_set['created_at']).replace('Z', '+00:00'))
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                ready_at = created + timedelta(hours=float(change_set.get('evaluation_hours') or 24))
            except (KeyError, TypeError, ValueError) as exc:
                raise HTTPException(status_code=500, detail='evaluation schedule is invalid') from exc
            if datetime.now(timezone.utc) < ready_at:
                raise HTTPException(status_code=409, detail=f'follow-up evidence is still being collected until {ready_at.isoformat()}')
            active_manager.events.update_calibration_change_set_status(change_set_id, 'reviewing')
        try:
            self._start_registered_ai_thread(
                'calibration_evaluation',
                self._run_calibration_evaluation,
                (change_set_id, active_config, active_manager),
                name=f'survng-calibration-evaluation-{change_set_id}',
            )
        except BaseException as exc:
            active_manager.events.update_calibration_evaluation(change_set_id, {'outcome': 'failed', 'error': f'Evaluation worker could not start: {redact_secret_text(exc)}'}, status='evaluation_failed')
            raise HTTPException(status_code=503, detail='calibration evaluation worker could not start') from exc
        return active_manager.events.get_calibration_change_set(change_set_id) or {}

    def calibration_rollback(self, change_set_id: int, request: CalibrationRollbackRequest) -> dict:
        if not request.confirmed:
            raise HTTPException(status_code=400, detail='explicit confirmation is required')
        with self.deps.manager_lock:
            active_events = self.deps.get_manager().events
            source = active_events.get_calibration_change_set(change_set_id)
            if source is None:
                raise HTTPException(status_code=404, detail='calibration change set not found')
            if source.get('action') != 'apply':
                raise HTTPException(status_code=400, detail='only applied calibration changes can be rolled back')
            already_rolled_back = active_events.calibration_rollback_change_ids(change_set_id)
            selected = []
            change_ids = set(request.change_ids)
            camera_ids = set(request.camera_ids)
            for item in source.get('changes') or []:
                if change_ids and str(item.get('id') or '') not in change_ids:
                    continue
                if camera_ids and str(item.get('camera_id') or '') not in camera_ids:
                    continue
                if str(item.get('id') or '') in already_rolled_back:
                    continue
                selected.append(item)
            if not selected:
                raise HTTPException(status_code=409 if already_rolled_back else 400, detail='the selected calibration changes are already rolled back' if already_rolled_back else 'no matching changes selected for rollback')
            conflicts = []
            inverse = []
            for item in selected:
                current = calibration_setting_value(self.deps.get_config(), scope=str(item.get('scope') or ''), camera_id=str(item.get('camera_id') or ''), setting=str(item.get('setting') or ''))
                if current != item.get('after'):
                    conflicts.append({'change_id': item.get('id'), 'setting': item.get('setting'), 'camera_id': item.get('camera_id'), 'expected': item.get('after'), 'current': current})
                inverse.append({'scope': item.get('scope'), 'camera_id': item.get('camera_id'), 'setting': item.get('setting'), 'proposed': item.get('before'), 'reason': f'Rollback of calibration change set {change_set_id}', 'source_change_id': item.get('id')})
            if conflicts and (not request.force_conflicts):
                raise HTTPException(status_code=409, detail={'message': 'newer configuration changes conflict with rollback', 'conflicts': conflicts})
            try:
                next_config, changes = apply_calibration_changes(self.deps.get_config(), inverse)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            before_config = self.deps.get_config().model_copy(deep=True)
            before_fingerprint = calibration_configuration_fingerprint(self.deps.get_config())
            _effective, apply_result = self.deps.apply_config_update(next_config)
            after_fingerprint = calibration_configuration_fingerprint(self.deps.get_config())
            try:
                rollback = active_events.create_calibration_change_set(run_id=source.get('run_id'), parent_change_set_id=change_set_id, action='rollback', status='rolled_back', evaluation_hours=24, configuration_fingerprint_before=before_fingerprint, configuration_fingerprint_after=after_fingerprint, changes=changes, apply_result={**apply_result, 'forced_conflicts': conflicts})
            except BaseException:
                try:
                    self.deps.apply_config_update(before_config)
                except Exception:
                    LOGGER.exception('calibration rollback ledger failure recovery was incomplete')
                raise
            rolled_count = len(already_rolled_back) + len(selected)
            total_count = len(source.get('changes') or [])
            active_events.update_calibration_change_set_status(change_set_id, 'rolled_back' if rolled_count >= total_count else 'partially_rolled_back')
        return {'ok': True, 'rollback': rollback, 'conflicts': conflicts}

    def _assistant_catalog(self, active_config: AppConfig, active_manager: AppManager) -> dict[str, Any]:
        face_store = getattr(active_manager, 'faces', None)
        try:
            people = face_store.people() if face_store is not None else []
        except Exception:
            LOGGER.warning('assistant catalog could not read recognized face names')
            people = []
        return {'cameras': [{'id': camera.id, 'name': camera.name} for camera in active_config.cameras], 'object_labels': list(active_manager.detector.labels), 'zones': sorted({zone.name for camera in active_config.cameras for zone in camera.zones if zone.enabled}), 'recognized_faces': [{'id': int(person.get('id') or 0), 'name': str(person.get('name') or '')[:128]} for person in people[:200] if str(person.get('name') or '').strip()]}

    def _assistant_system_evidence(self, active_manager: AppManager) -> AssistantEvidence:
        status = self.deps.system_telemetry.system_status(active_manager)
        cameras = active_manager.statuses()
        unhealthy = [{'camera_id': str(camera.get('id') or ''), 'running': bool(camera.get('running')), 'connected': bool(camera.get('connected')), 'frame_fresh': bool(camera.get('frame_fresh')), 'recording': bool(camera.get('recording') or camera.get('sub_recording')), 'last_frame_age_seconds': camera.get('last_frame_age_seconds'), 'last_error': str(camera.get('last_error') or '')[:500]} for camera in cameras if not camera.get('running') or not camera.get('frame_fresh') or (not (camera.get('recording') or camera.get('sub_recording')))]
        detector = status.get('detector') or {}
        runtime = detector.get('runtime') or {}
        retention = active_manager.recorder.retention_status()
        retention_plan = retention.get('plan') or {}
        retention_reclaim = retention_plan.get('reclaim') or {}
        last_retention_run = retention.get('last_run') or {}
        payload = {'cameras': status.get('cameras'), 'unhealthy_cameras': unhealthy, 'storage': status.get('storage'), 'detector': {'enabled': detector.get('enabled'), 'backend': detector.get('loaded_backend'), 'device': detector.get('loaded_device'), 'ready': bool(detector.get('openvino_loaded') or detector.get('coreml_loaded')), 'average_inference_ms': runtime.get('average_inference_ms'), 'queue_depth': runtime.get('queue_depth'), 'failed_inferences': runtime.get('failed_inferences'), 'active_workers': runtime.get('active_workers'), 'configured_workers': runtime.get('configured_workers'), 'reid_ready': bool((detector.get('reid') or {}).get('loaded'))}, 'mqtt': {key: (status.get('mqtt') or {}).get(key) for key in ('enabled', 'connected', 'last_error', 'publish_failures', 'pending_incidents', 'server_lifecycle')}, 'go2rtc': status.get('go2rtc'), 'retention': {'state': retention.get('state'), 'enabled': retention.get('enabled'), 'automatic_cleanup': retention.get('automatic_cleanup'), 'last_plan_at': retention.get('last_plan_at'), 'last_run_at': retention.get('last_run_at'), 'error': retention.get('error'), 'planned_reclaim_bytes': retention_reclaim.get('planned_bytes'), 'last_deleted_files': last_retention_run.get('deleted_files'), 'last_deleted_bytes': last_retention_run.get('deleted_bytes')}}
        total = int((status.get('cameras') or {}).get('total') or 0)
        online = int((status.get('cameras') or {}).get('online') or 0)
        return AssistantEvidence(evidence_id='E-system', kind='system_health', title='Current SurvNG health', summary=f'{online}/{total} cameras online; {len(unhealthy)} cameras need attention.', data=payload, href='/config#telemetry')

    def _assistant_camera_evidence(self, active_manager: AppManager, camera_id: str) -> list[AssistantEvidence]:
        requested = camera_id.strip().lower()
        evidence: list[AssistantEvidence] = []
        for camera in active_manager.statuses():
            current_id = str(camera.get('id') or '')
            if requested and current_id.lower() != requested:
                continue
            motion = camera.get('motion_qualification') or {}
            tracking = camera.get('object_tracking') or {}
            data = {'camera_id': current_id, 'name': camera.get('name') or current_id, 'running': bool(camera.get('running')), 'connected': bool(camera.get('connected')), 'frame_fresh': bool(camera.get('frame_fresh')), 'last_frame_age_seconds': camera.get('last_frame_age_seconds'), 'recording': bool(camera.get('recording')), 'sub_recording': bool(camera.get('sub_recording')), 'detection_enabled': bool(camera.get('detection_enabled')), 'onvif': {'enabled': bool(camera.get('onvif_enabled')), 'connected': bool(camera.get('onvif_connected')), 'notifications': int(camera.get('onvif_notifications_received') or 0), 'motion_events': int(camera.get('onvif_motion_events_received') or 0), 'poll_errors': int(camera.get('onvif_poll_errors') or 0), 'poll_timeouts': int(camera.get('onvif_poll_timeouts') or 0), 'last_motion_at': camera.get('onvif_last_motion_event_at'), 'last_error': str(camera.get('onvif_last_error') or '')[:500]}, 'motion': {key: motion.get(key) for key in ('mode', 'sensitivity', 'triggers', 'passed', 'audit_rejected', 'suppressed', 'dropped_triggers', 'queue_depth', 'visual_backup_triggers', 'visual_backup_not_ready', 'visual_backup_uncorrelated_objects')}, 'tracking': {key: tracking.get(key) for key in ('active', 'frames_processed', 'track_count', 'reid_attempts', 'reid_successes', 'reid_failures', 'coverage_incomplete')}}
            healthy = data['running'] and data['frame_fresh']
            evidence.append(AssistantEvidence(evidence_id=f'E-camera-{current_id}', kind='camera_health', title=str(data['name']), summary=f'{current_id} is {('healthy' if healthy else 'not healthy')}; recording is {('active' if data['recording'] or data['sub_recording'] else 'inactive')}.', data=data, href=f'/?camera={quote(current_id, safe='')}'))
        return evidence

    def _assistant_configuration_evidence(self, active_config: AppConfig) -> AssistantEvidence:
        assistant_provider = AssistantProvider(active_config.audit_ai)
        data = {'ai': {'enabled': active_config.audit_ai.enabled, 'assistant_enabled': active_config.audit_ai.assistant_enabled, 'provider': active_config.audit_ai.provider, 'analysis_and_fast_model': assistant_provider.model_for_tier('fast'), 'deep_reasoning_model': assistant_provider.model_for_tier('deep'), 'deep_reasoning_uses_separate_model': assistant_provider.model_for_tier('deep') != assistant_provider.model_for_tier('fast'), 'assistant_read_only': False, 'supported_actions': ['create_media_export']}, 'recording': {'segment_seconds': active_config.recording_segment_seconds, 'cache_max_gb': active_config.recording_cache_max_gb, 'cache_max_days': active_config.recording_cache_max_days, 'prewarm': active_config.recording_cache_prewarm, 'retention': active_config.retention.model_dump(mode='json')}, 'motion': active_config.motion_qualification.model_dump(mode='json'), 'detector': {'enabled': active_config.detector.enabled, 'backend': active_config.detector.backend, 'device': active_config.detector.device, 'confidence_threshold': active_config.detector.confidence_threshold, 'nms_threshold': active_config.detector.nms_threshold, 'event_confirmation_frames': active_config.detector.event_confirmation_frames, 'event_class_confirmation_frames': active_config.detector.event_class_confirmation_frames, 'event_class_confidence_thresholds': active_config.detector.event_class_confidence_thresholds, 'zone_only_incident_eligibility': active_config.detector.require_incident_zone, 'tracking': active_config.detector.tracking.model_dump(mode='json')}, 'mqtt': {'enabled': active_config.mqtt.enabled, 'tls': active_config.mqtt.tls, 'discovery_enabled': active_config.mqtt.discovery_enabled, 'incident_events_enabled': active_config.mqtt.incident_events_enabled, 'server_status_enabled': active_config.mqtt.server_status_enabled}, 'cameras': [{'id': camera.id, 'name': camera.name, 'record': camera.record, 'record_sub': camera.record_sub, 'retention': camera.retention.model_dump(mode='json'), 'zone_only_incident_eligibility': camera.require_incident_zone, 'motion': camera.motion_qualification.model_dump(mode='json'), 'onvif_enabled': camera.onvif.enabled, 'zone_names': [zone.name for zone in camera.zones if zone.enabled]} for camera in active_config.cameras]}
        return AssistantEvidence(evidence_id='E-config', kind='configuration', title='Active safe configuration', summary=f'Credential-free configuration for {len(active_config.cameras)} cameras.', data=data, href='/config')

    def _assistant_event_objects(self, event: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        objects: list[dict[str, Any]] = []
        qualification: dict[str, Any] = {}
        tracking: dict[str, Any] = {}
        for item in event.get('objects') or []:
            if not isinstance(item, dict):
                continue
            if item.get('label'):
                objects.append({key: item.get(key) for key in (
                    'label', 'confidence', 'incident_eligible', 'zones', 'box',
                    'semantic_tier', 'semantic_rescue_threshold',
                    'semantic_standard_threshold', 'semantic_rescue_admitted',
                    'incident_admission_reason', 'temporal_observations',
                    'temporal_required_observations',
                    'temporal_center_displacement_ratio',
                    'temporal_center_path_ratio',
                    'temporal_peak_confidence_offset_seconds',
                    'temporal_robust_new_appearance',
                    'temporal_zone_entry', 'activity_role', 'motion_correlated',
                    'motion_correlation', 'track_id', 'track_state',
                )})
            elif isinstance(item.get('motion_qualification'), dict):
                qualification = item['motion_qualification']
            elif isinstance(item.get('object_tracking'), dict):
                tracking = item['object_tracking']
        return (objects, qualification, tracking)

    def _assistant_incident_payload(self, incident: dict[str, Any]) -> dict[str, Any]:
        events = []
        for event in incident.get('events') or []:
            objects, qualification, tracking = self._assistant_event_objects(event)
            events.append({'id': event.get('id'), 'created_at': event.get('created_at'), 'kind': event.get('kind'), 'topic': event.get('topic'), 'trigger_source': event.get('trigger_source'), 'objects': objects, 'motion_qualification': qualification, 'object_tracking': tracking, 'recording_available': bool(event.get('recording_path')), 'faces': event.get('faces') or []})
        return {'incident_id': incident.get('id'), 'representative_event_id': incident.get('representative_event_id'), 'camera_id': incident.get('camera_id'), 'start_at': incident.get('start_at'), 'end_at': incident.get('end_at'), 'duration_seconds': incident.get('duration_seconds'), 'event_count': incident.get('event_count'), 'trigger_source': incident.get('trigger_source'), 'labels': incident.get('labels') or [], 'zones': incident.get('zones') or [], 'motion_observations': [{key: observation.get(key) for key in ('id', 'created_at', 'category', 'reason', 'score', 'threshold', 'object_detected', 'trigger_count', 'interpretation')} for observation in incident.get('motion_observations') or [] if isinstance(observation, dict)], 'events': events}

    def _assistant_incident_evidence(self, incident: dict[str, Any], evidence_event_id: int | None=None) -> AssistantEvidence:
        payload = self._assistant_incident_payload(incident)
        event_ids = [str(event.get('id')) for event in incident.get('events') or [] if event.get('id')]
        query = quote(','.join(event_ids), safe=',')
        event_id = evidence_event_id or int(incident.get('representative_event_id') or (event_ids[0] if event_ids else 0))
        image_event_id = int(incident.get('representative_event_id') or event_id)
        return AssistantEvidence(evidence_id=f'E-incident-{event_id}', kind='incident', title=f'{incident.get('camera_id')} · {incident.get('start_at')}', summary=f'{len(payload['events'])} event(s); labels: {(', '.join(payload['labels']) if payload['labels'] else 'motion only')}.', data=payload, href=f'/incidents?event_ids={query}', image_url=f'/api/events/{image_event_id}/thumbnail.jpg?width=960&quality=82' if image_event_id > 0 else '')

    def _assistant_inspect_incident(self, event_id: int, active_manager: AppManager) -> AssistantEvidence | None:
        incident = self.deps.incident_queries.resolve_event(active_manager, event_id)
        if incident is None:
            return None
        return self._assistant_incident_evidence(incident, event_id)

    def _assistant_motion_change_current_value(self, active_config: AppConfig, camera: CameraConfig, change: AuditAiChange) -> object:
        if change.scope == 'global':
            return getattr(active_config.motion_qualification, change.setting)
        override = getattr(camera.motion_qualification, change.setting)
        if override is None or override == 'inherit':
            return getattr(active_config.motion_qualification, change.setting)
        return override

    def _assistant_motion_change_previews(self, active_config: AppConfig, camera: CameraConfig, changes: list[AuditAiChange]) -> tuple[list[AuditAiChange], list[dict[str, Any]]]:
        unique: list[AuditAiChange] = []
        previews: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for change in changes:
            key = (change.scope, change.setting)
            if key in seen:
                continue
            seen.add(key)
            proposed = validate_tuning_value(change.setting, change.value)
            current = self._assistant_motion_change_current_value(active_config, camera, change)
            if current == proposed:
                continue
            normalized = change.model_copy(update={'value': proposed})
            unique.append(normalized)
            previews.append({'scope': change.scope, 'setting': change.setting, 'current': current, 'proposed': proposed, 'reason': change.reason})
        return (unique, previews)

    def _assistant_motion_config_fingerprint(self, active_config: AppConfig, camera: CameraConfig) -> str:
        payload = {'global': active_config.motion_qualification.model_dump(mode='json'), 'camera': camera.motion_qualification.model_dump(mode='json')}
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(encoded.encode('utf-8')).hexdigest()

    def _ai_recommendation_payload(self, *, kind: str, record_id: int, camera_id: str, configuration_fingerprint: str, changes: list[AuditAiChange], issued_at: int) -> bytes:
        normalized = [{'scope': change.scope, 'setting': change.setting, 'value': validate_tuning_value(change.setting, change.value), 'reason': change.reason} for change in changes]
        payload = {'version': 1, 'kind': kind, 'record_id': int(record_id), 'camera_id': camera_id, 'configuration_fingerprint': configuration_fingerprint, 'changes': normalized, 'issued_at': int(issued_at)}
        return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

    def _issue_ai_recommendation_token(self, *, kind: str, record_id: int, camera_id: str, configuration_fingerprint: str, changes: list[AuditAiChange]) -> str:
        issued_at = int(time.time())
        signature = hmac.new(AI_RECOMMENDATION_SECRET, self._ai_recommendation_payload(kind=kind, record_id=record_id, camera_id=camera_id, configuration_fingerprint=configuration_fingerprint, changes=changes, issued_at=issued_at), hashlib.sha256).hexdigest()
        return f'v1.{issued_at}.{signature}'

    def _verify_ai_recommendation_token(self, token: str, *, kind: str, record_id: int, camera_id: str, configuration_fingerprint: str, changes: list[AuditAiChange]) -> bool:
        try:
            version, raw_issued_at, supplied_signature = token.split('.', 2)
            issued_at = int(raw_issued_at)
        except (AttributeError, TypeError, ValueError):
            return False
        now = int(time.time())
        if version != 'v1' or issued_at > now + 30 or now - issued_at > AI_RECOMMENDATION_MAX_AGE_SECONDS:
            return False
        expected = hmac.new(AI_RECOMMENDATION_SECRET, self._ai_recommendation_payload(kind=kind, record_id=record_id, camera_id=camera_id, configuration_fingerprint=configuration_fingerprint, changes=changes, issued_at=issued_at), hashlib.sha256).hexdigest()
        return hmac.compare_digest(supplied_signature, expected)

    def _assistant_visual_incident_evidence(self, event_id: int, active_config: AppConfig, active_manager: AppManager) -> AssistantEvidence | None:
        incident = self.deps.incident_queries.resolve_event(active_manager, event_id)
        if incident is None:
            return None
        camera = camera_by_id(active_config, str(incident.get('camera_id') or ''))
        if camera is None:
            return None
        candidate_ids = list(dict.fromkeys([int(incident.get('representative_event_id') or 0), *[int(event.get('id') or 0) for event in incident.get('events') or []]]))
        snapshot_path = None
        source_event_id = 0
        for candidate_id in candidate_ids:
            if candidate_id <= 0:
                continue
            raw_event = active_manager.events.get(candidate_id)
            if raw_event is None:
                continue
            try:
                snapshot_path = event_snapshot_path(active_manager.storage_dir, raw_event)
                source_event_id = candidate_id
                break
            except (FileNotFoundError, PermissionError):
                continue
        if snapshot_path is None:
            raise AuditAiError('incident has no retained image available for visual review')
        config_evidence = self._assistant_configuration_evidence(active_config).data
        camera_configuration = next((item for item in config_evidence.get('cameras', []) if item.get('id') == camera.id), {})
        camera_evidence = self._assistant_camera_evidence(active_manager, camera.id)
        motion_override = camera.motion_qualification
        motion_graphs = resolve_motion_pipeline_graphs(
            active_config.motion_qualification,
            motion_override,
        )
        effective_mode = (
            active_config.motion_qualification.mode
            if motion_override.mode == 'inherit'
            else motion_override.mode
        )
        require_incident_zone = (
            active_config.detector.require_incident_zone
            if camera.require_incident_zone is None
            else camera.require_incident_zone
        )
        context = {
            'motion_paradigm': motion_paradigm_context(
                mode=effective_mode,
                onvif_enabled=camera.onvif.enabled,
                has_live_substream=bool(camera.live_stream_url),
                fusion=guided_fusion_settings(motion_graphs.fusion),
                require_incident_zone=require_incident_zone,
            ),
            'incident': self._assistant_incident_payload(incident),
            'camera_health': camera_evidence[0].data if camera_evidence else {},
            'active_configuration': {
                'global_motion': config_evidence.get('motion') or {},
                'detector': config_evidence.get('detector') or {},
                'camera': camera_configuration,
            },
            'image_source_event_id': source_event_id,
            'limitations': [
                'The image is one representative moment, not the full recording.',
                'Movement and causal correlation require temporal telemetry; the image alone cannot prove them.',
                'Tracking begins after the initial object decision.',
                'Only bounded motion settings may be proposed from this review.',
            ],
        }
        advice = IncidentVisualReviewer(active_config.audit_ai).review(snapshot_path, context)
        changes, previews = self._assistant_motion_change_previews(active_config, camera, [change for change in advice.changes if change.scope == 'camera'])
        advice_payload = advice.model_dump(mode='json')
        advice_payload['changes'] = [change.model_dump(mode='json') for change in changes]
        configuration_fingerprint = self._assistant_motion_config_fingerprint(active_config, camera)
        details = {'event_id': event_id, 'source_event_id': source_event_id, 'camera_id': camera.id, 'advice': advice_payload, 'proposals': previews, 'can_apply': bool(previews and active_config.audit_ai.allow_apply_recommendations), 'apply_requires_confirmation': True, 'configuration_fingerprint': configuration_fingerprint, 'recommendation_proof': self._issue_ai_recommendation_token(kind='incident_visual', record_id=event_id, camera_id=camera.id, configuration_fingerprint=configuration_fingerprint, changes=changes)}
        incident_evidence = self._assistant_incident_evidence(incident, event_id)
        return AssistantEvidence(evidence_id=f'E-visual-{event_id}', kind='incident_visual_review', title=f'Visual review · {camera.name}', summary=f'{advice.verdict.replace('_', ' ')} ({round(advice.confidence * 100)}% confidence); {len(previews)} bounded setting proposal(s).', data={**{key: value for key, value in details.items() if key != 'recommendation_proof'}, 'incident_evidence': incident_evidence.data}, href=incident_evidence.href, image_url=f'/api/events/{source_event_id}/thumbnail.jpg?width=960&quality=82', client_data=details)

    def _assistant_parse_datetime(self, value: str, selected_zone: ZoneInfo) -> datetime | None:
        if not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=selected_zone)
        return parsed.astimezone(timezone.utc)

    def _assistant_media_export_evidence(self, call: AssistantToolCall, request: AssistantChatRequest, active_config: AppConfig) -> AssistantEvidence:
        try:
            selected_zone = ZoneInfo(request.context.time_zone)
        except (ZoneInfoNotFoundError, ValueError):
            selected_zone = ZoneInfo('America/New_York')
        export_kind = call.export_kind
        camera_id = call.camera_id or request.context.camera_id
        camera = camera_by_id(active_config, camera_id) if camera_id else None
        start = self._assistant_parse_datetime(call.start_at, selected_zone)
        end = self._assistant_parse_datetime(call.end_at, selected_zone)
        questions: list[str] = []
        suggestions: list[str] = []
        if not export_kind:
            questions.append('Should this be a normal video clip or a timelapse?')
            suggestions.extend(['Make it a timelapse', 'Make it a normal video clip'])
        if camera is None:
            if camera_id:
                questions.append(f"I couldn't find a camera named {camera_id}. Which camera should I use?")
            else:
                questions.append('Which camera should I use?')
            suggestions.extend((camera.name for camera in active_config.cameras[:4]))
        if start is None and end is None:
            questions.append('What date and start/end times should the export cover?')
        elif start is None:
            questions.append('What time should the export start?')
        elif end is None:
            questions.append('What time should the export end?')
        if questions:
            question = ' '.join(questions)
            return AssistantEvidence(evidence_id='E-export-clarification', kind='media_export_clarification', title='More export details needed', summary=question, data={'questions': questions, 'suggestions': list(dict.fromkeys(suggestions))[:4]}, href='/recordings', client_data={'questions': questions})
        assert camera is not None and start is not None and (end is not None)
        if end <= start:
            return AssistantEvidence(evidence_id='E-export-clarification', kind='media_export_clarification', title='Export times need correction', summary='The end time must be after the start time. What start and end times should I use?', data={'questions': ['What corrected start and end times should I use?'], 'suggestions': []}, href='/recordings')
        maximum = timedelta(days=1 if export_kind == 'recording' else 7)
        if end - start > maximum:
            label = '24 hours' if export_kind == 'recording' else '7 days'
            return AssistantEvidence(evidence_id='E-export-clarification', kind='media_export_clarification', title='Export range is too long', summary=f'That {export_kind} exceeds the {label} limit. What shorter range should I use?', data={'questions': [f'What range within {label} should I use?'], 'suggestions': []}, href='/recordings')
        source = recording_source(call.source or 'main')
        options: dict[str, object] = {'height': call.height or 0}
        if export_kind == 'timelapse':
            options = {'sample_interval_seconds': call.sample_interval_seconds or 30.0, 'output_fps': call.output_fps or 30, **({'height': call.height or 720} if call.height is not None or call.width is None else {'width': call.width})}
        try:
            job = self.deps.media_export_manager().create({'kind': export_kind, 'camera_id': camera.id, 'source': source, 'start_epoch': start.timestamp(), 'end_epoch': end.timestamp(), 'options': options, 'origin': 'assistant'})
        except RuntimeError as exc:
            LOGGER.warning('assistant could not queue media export: %s', exc)
            return AssistantEvidence(evidence_id='E-export-clarification', kind='media_export_clarification', title='Export could not be queued', summary='The export queue is unavailable right now. Please try again shortly.', data={'questions': [], 'suggestions': ['Try the export again']}, href='/recordings')
        job_id = str(job['id'])
        local_start = start.astimezone(selected_zone)
        local_end = end.astimezone(selected_zone)
        kind_label = 'timelapse' if export_kind == 'timelapse' else 'video clip'
        summary = f'Queued a {kind_label} for {camera.name} from {local_start.strftime('%b %-d, %-I:%M %p')} to {local_end.strftime('%b %-d, %-I:%M %p')}.'
        return AssistantEvidence(evidence_id=f'E-export-{job_id[:12]}', kind='media_export_job', title=f'{camera.name} {kind_label}', summary=summary, data={'job_id': job_id, 'kind': export_kind, 'camera_id': camera.id, 'source': source, 'start_at': start.isoformat(), 'end_at': end.isoformat(), 'status': job.get('status'), 'options': options}, href=f'/recordings?camera={quote(camera.id, safe='')}&at={start.timestamp():.3f}&source={source}', client_data={'media_export': {'id': job_id, 'kind': export_kind, 'camera_id': camera.id, 'source': source, 'start_at': start.isoformat(), 'end_at': end.isoformat(), 'status': str(job.get('status') or 'queued'), 'phase': str(job.get('phase') or 'Queued'), 'progress': float(job.get('progress') or 0)}})

    def _assistant_media_export_answer(self, evidence: AssistantEvidence) -> AssistantAnswer:
        if evidence.kind == 'media_export_clarification':
            return AssistantAnswer(answer=f'{evidence.summary} [{evidence.evidence_id}]', citations=[evidence.evidence_id], suggestions=list(evidence.data.get('suggestions') or [])[:4])
        return AssistantAnswer(answer=f'I started the requested export. It will appear here when the MP4 is ready, and you can leave this panel open while it runs. [{evidence.evidence_id}]', citations=[evidence.evidence_id], suggestions=[])

    def _assistant_search_incidents(self, call: AssistantToolCall, time_zone: str, active_manager: AppManager) -> list[AssistantEvidence]:
        try:
            selected_zone = ZoneInfo(time_zone)
        except (ZoneInfoNotFoundError, ValueError):
            selected_zone = ZoneInfo('America/New_York')
        now = datetime.now(timezone.utc)
        start = self._assistant_parse_datetime(call.start_at, selected_zone) or now - timedelta(hours=24)
        end = self._assistant_parse_datetime(call.end_at, selected_zone) or now
        if end <= start:
            start, end = (end, start)
        start = max(start, end - timedelta(days=31))
        rows = [_event_row(row) for row in active_manager.events.between_compact(start.isoformat(), end.isoformat())]
        summaries = _filter_incident_summaries(_incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS), call.event_type, call.camera_id, call.object_label, call.zone)
        candidate_summaries = summaries[:min(250, max(call.limit * 8, call.limit))]
        hydrated = self.deps.incident_queries.with_faces(active_manager, self.deps.incident_queries.hydrate(active_manager, candidate_summaries))
        filtered: list[dict[str, Any]] = []
        wanted_face = call.face_name.strip().lower()
        for incident in hydrated:
            payload = self._assistant_incident_payload(incident)
            detections = [obj for event in payload['events'] for obj in event['objects'] if obj.get('incident_eligible') is not False]
            if call.minimum_confidence is not None and (not any((float(obj.get('confidence') or 0) >= call.minimum_confidence for obj in detections))):
                continue
            if wanted_face:
                face_names = {str(face.get('name') or '').strip().lower() for event in payload['events'] for face in event.get('faces') or [] if isinstance(face, dict)}
                if wanted_face not in face_names:
                    continue
            filtered.append(incident)
            if len(filtered) >= call.limit:
                break
        query_summary = AssistantEvidence(evidence_id='E-search', kind='incident_search', title='Incident search', summary=f'Found {len(filtered)} matching incident(s) in the searched window.', data={'start_at': start.isoformat(), 'end_at': end.isoformat(), 'camera_id': call.camera_id, 'event_type': call.event_type, 'object_label': call.object_label, 'zone': call.zone, 'minimum_confidence': call.minimum_confidence, 'face_name': call.face_name, 'returned': len(filtered), 'candidate_count': len(summaries), 'scanned_candidates': len(candidate_summaries), 'candidate_scan_limited': len(candidate_summaries) < len(summaries)}, href='/incidents')
        evidence = [query_summary]
        for incident in filtered:
            event_id = int(incident.get('representative_event_id') or 0)
            if event_id:
                evidence.append(self._assistant_incident_evidence(incident, event_id))
        return evidence

    def _assistant_semantic_search(self, call: AssistantToolCall, time_zone: str, active_manager: AppManager) -> list[AssistantEvidence]:
        query = call.query.strip()
        if not query:
            return []
        try:
            selected_zone = ZoneInfo(time_zone)
        except (ZoneInfoNotFoundError, ValueError):
            selected_zone = ZoneInfo('America/New_York')
        now = datetime.now(timezone.utc)
        start = self._assistant_parse_datetime(call.start_at, selected_zone) or now - timedelta(hours=24)
        end = self._assistant_parse_datetime(call.end_at, selected_zone) or now
        if end <= start:
            start, end = (end, start)
        start = max(start, end - timedelta(days=31))
        try:
            hits = active_manager.semantic_search.search_text(query, camera_ids=[call.camera_id] if call.camera_id else [], object_labels=[call.object_label] if call.object_label else [], start_at=start.isoformat(), end_at=end.isoformat(), limit=min(call.limit * 4, 200), minimum_score=-1.0)
        except RuntimeError as exc:
            return [AssistantEvidence(evidence_id='E-semantic-status', kind='semantic_search_status', title='Visual search unavailable', summary=str(exc), data=active_manager.semantic_search_status(), href='/recordings/search')]
        best: dict[int, Any] = {}
        for hit in hits:
            best.setdefault(hit.event_id, hit)
            if len(best) >= call.limit:
                break
        evidence = [AssistantEvidence(evidence_id='E-semantic-search', kind='semantic_search', title=f'Visual search · “{query}”', summary=f'Found {len(best)} visually similar incident(s) in the indexed evidence.', data={'query': query, 'start_at': start.isoformat(), 'end_at': end.isoformat(), 'camera_id': call.camera_id, 'object_label': call.object_label, 'matches': [{'event_id': hit.event_id, 'score': round(hit.score, 4), 'source_kind': hit.source_kind, 'object_label': hit.object_label} for hit in best.values()], 'limitations': ['Similarity is not identity proof.', 'Search currently covers indexed object incidents, not every recording frame.']}, href='/recordings/search')]
        for event_id in best:
            incident = self.deps.incident_queries.resolve_event(active_manager, event_id)
            if incident is not None:
                evidence.append(self._assistant_incident_evidence(incident, event_id))
        return evidence

    def _assistant_recent_activity_summary(self, call: AssistantToolCall, time_zone: str, active_manager: AppManager) -> AssistantEvidence:
        try:
            selected_zone = ZoneInfo(time_zone)
        except (ZoneInfoNotFoundError, ValueError):
            selected_zone = ZoneInfo('America/New_York')
        now = datetime.now(timezone.utc)
        start = self._assistant_parse_datetime(call.start_at, selected_zone) or now - timedelta(hours=24)
        end = self._assistant_parse_datetime(call.end_at, selected_zone) or now
        if end <= start:
            start, end = (end, start)
        start = max(start, end - timedelta(days=31))
        rows = [_event_row(row) for row in active_manager.events.between_compact(start.isoformat(), end.isoformat())]
        summaries = _filter_incident_summaries(_incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS), 'object', call.camera_id, call.object_label, call.zone)

        def counts(values: list[str]) -> dict[str, int]:
            result: dict[str, int] = {}
            for value in values:
                if value:
                    result[value] = result.get(value, 0) + 1
            return dict(sorted(result.items(), key=lambda item: (-item[1], item[0])))
        duration_minutes = max(0, round((end - start).total_seconds() / 60))
        camera_ids = {str(item.get('camera_id') or '') for item in summaries}
        recent = [{'event_id': int(item.get('representative_event_id') or 0), 'camera_id': str(item.get('camera_id') or ''), 'started_at': item.get('start_at'), 'labels': list(item.get('labels') or []), 'zones': list(item.get('zones') or []), 'trigger_source': item.get('trigger_source') or 'camera'} for item in summaries[:8]]
        return AssistantEvidence(evidence_id='E-activity', kind='recent_activity_summary', title='Recent activity', summary=f'{len(summaries)} incidents across {len(camera_ids - {''})} cameras during the last {duration_minutes} minutes.', data={'start_at': start.isoformat(), 'end_at': end.isoformat(), 'duration_minutes': duration_minutes, 'incident_count': len(summaries), 'object_incident_count': len(summaries), 'camera_counts': counts([str(item.get('camera_id') or '') for item in summaries]), 'object_label_counts': counts([str(label) for item in summaries for label in item.get('labels') or []]), 'zone_counts': counts([str(zone) for item in summaries for zone in item.get('zones') or []]), 'trigger_counts': counts([str(item.get('trigger_source') or 'camera') for item in summaries]), 'recent_notable_incidents': recent, 'filters': {'camera_id': call.camera_id, 'event_type': 'object', 'object_label': call.object_label, 'zone': call.zone}}, href='/incidents')

    def _assistant_activity_followups(self, evidence: AssistantEvidence) -> list[str]:
        data = evidence.data
        minutes = max(1, int(data.get('duration_minutes') or 0))
        if minutes % 1440 == 0:
            count = minutes // 1440
            period = f'last {count} day{('s' if count != 1 else '')}'
        elif minutes % 60 == 0:
            count = minutes // 60
            period = f'last {count} hour{('s' if count != 1 else '')}'
        else:
            period = f'last {minutes} minutes'
        followups: list[str] = []
        camera_counts = data.get('camera_counts') or {}
        if camera_counts:
            busiest = next(iter(camera_counts))
            followups.append(f'What happened on {busiest} in the {period}?')
        if int(data.get('object_incident_count') or 0):
            followups.append(f'Show me the object incidents from the {period}')
        triggers = data.get('trigger_counts') or {}
        if any((key in triggers for key in ('adaptive', 'visual_backup', 'adaptive/visual_backup'))):
            followups.append(f'Which incidents did the visual motion check rescue in the {period}?')
        return followups[:3]

    def _assistant_prioritize_trace_candidates(self, candidate_summaries: list[dict[str, Any]], appearance_matches: list[dict[str, Any]], appearance_event_ids: set[int], distance_from_anchor: Callable[[dict[str, Any]], float], *, limit: int=500) -> list[dict[str, Any]]:
        """Keep strongest appearance evidence before filling a bounded temporal scan."""
        candidate_summaries = sorted(candidate_summaries, key=distance_from_anchor)
        appearance_summaries = [summary for summary in candidate_summaries if any((int(event.get('id') or 0) in appearance_event_ids for event in summary.get('events') or []))]
        appearance_score_by_event = {int(item.get('event_id') or 0): float(item.get('similarity') or 0.0) for item in appearance_matches if item.get('visually_similar')}
        appearance_summaries.sort(key=lambda summary: max((appearance_score_by_event.get(int(event.get('id') or 0), 0.0) for event in summary.get('events') or []), default=0.0), reverse=True)
        retained_ids = {id(summary) for summary in appearance_summaries}
        retained_appearance = appearance_summaries[:min(100, limit)]
        return retained_appearance + [summary for summary in candidate_summaries if id(summary) not in retained_ids][:max(0, limit - len(retained_appearance))]

    def _assistant_trace_across_cameras(self, call: AssistantToolCall, request: AssistantChatRequest, active_manager: AppManager) -> list[AssistantEvidence]:
        event_id = call.event_id or request.context.incident_event_id
        if not event_id and (not call.face_name.strip()) and (not call.object_label.strip()):
            return []
        anchor = self.deps.incident_queries.resolve_event(active_manager, int(event_id)) if event_id else None
        if event_id and anchor is None:
            return []
        try:
            selected_zone = ZoneInfo(request.context.time_zone)
        except (ZoneInfoNotFoundError, ValueError):
            selected_zone = ZoneInfo('America/New_York')
        now = datetime.now(timezone.utc)
        try:
            anchor_at = datetime.fromisoformat(str((anchor or {}).get('start_at') or '').replace('Z', '+00:00'))
        except ValueError:
            anchor_at = now
        if anchor_at.tzinfo is None:
            anchor_at = anchor_at.replace(tzinfo=timezone.utc)
        default_start = anchor_at - timedelta(minutes=15) if anchor else now - timedelta(hours=24)
        default_end = anchor_at + timedelta(minutes=15) if anchor else now
        start = self._assistant_parse_datetime(call.start_at, selected_zone) or default_start
        end = self._assistant_parse_datetime(call.end_at, selected_zone) or default_end
        if end <= start:
            start, end = (end, start)
        start = max(start, end - timedelta(hours=24))
        appearance_matches: list[dict[str, Any]] = []
        appearance_index = getattr(active_manager, 'appearance_index', None)
        if event_id and appearance_index is not None:
            try:
                appearance_matches = appearance_index.matches(int(event_id), start_at=start.isoformat(), end_at=end.isoformat(), cross_camera_only=True, limit=100)
            except Exception:
                LOGGER.exception('cross-camera appearance lookup failed for event %s', event_id)
        appearance_event_ids = {int(item.get('event_id') or 0) for item in appearance_matches if item.get('visually_similar')}
        rows = [_event_row(row) for row in active_manager.events.between_compact(start.isoformat(), end.isoformat())]
        summaries = _incident_rows(rows, DEFAULT_INCIDENT_GAP_SECONDS)
        wanted_label = call.object_label.strip().lower()
        anchor_labels = {str(label).strip().lower() for label in (anchor or {}).get('labels') or [] if str(label).strip()}
        target_labels = {wanted_label} if wanted_label else anchor_labels
        candidate_summaries = [summary for summary in summaries if not target_labels or target_labels & {str(label).strip().lower() for label in summary.get('labels') or []} or bool(call.face_name.strip()) or any((int(event.get('id') or 0) in appearance_event_ids for event in summary.get('events') or []))]

        def distance_from_anchor(summary: dict[str, Any]) -> float:
            parsed = self._assistant_parse_datetime(str(summary.get('start_at') or ''), selected_zone)
            return abs(parsed.timestamp() - anchor_at.timestamp()) if parsed is not None else float('inf')
        candidate_summaries = self._assistant_prioritize_trace_candidates(candidate_summaries, appearance_matches, appearance_event_ids, distance_from_anchor)
        candidates = self.deps.incident_queries.with_faces(active_manager, self.deps.incident_queries.hydrate(active_manager, candidate_summaries))
        correlation_anchor = anchor or {'representative_event_id': 0, 'camera_id': '', 'start_at': start.isoformat(), 'labels': [call.object_label] if call.object_label else [], 'faces': []}
        matches = correlate_incident_timeline(correlation_anchor, candidates, object_label=call.object_label, face_name=call.face_name, limit=min(call.limit, 12))
        incident_by_event_id: dict[int, dict[str, Any]] = {}
        for incident in candidates:
            representative_id = int(incident.get('representative_event_id') or 0)
            if representative_id > 0:
                incident_by_event_id[representative_id] = incident
            for event in incident.get('events') or []:
                candidate_event_id = int(event.get('id') or 0)
                if candidate_event_id > 0:
                    incident_by_event_id[candidate_event_id] = incident
        matches_by_event_id = {int(item.get('event_id') or 0): item for item in matches}
        for appearance in appearance_matches:
            if not appearance.get('visually_similar'):
                continue
            matched_incident = incident_by_event_id.get(int(appearance['event_id']))
            if matched_incident is None:
                continue
            representative_id = int(matched_incident.get('representative_event_id') or appearance['event_id'])
            similarity = float(appearance.get('similarity') or 0.0)
            reason = f'{str(appearance.get('model_kind') or 'object').title()} appearance is {round(similarity * 100)}% similar using the same ReID model'
            existing = matches_by_event_id.get(representative_id)
            if existing is not None:
                existing.setdefault('reasons', []).append(reason)
                existing['appearance_similarity'] = round(similarity, 4)
                if existing.get('match_strength') not in {'confirmed_identity', 'possible_identity'}:
                    existing['match_strength'] = 'appearance_similarity'
                    existing['confidence'] = round(similarity, 3)
                continue
            matched_at = str(matched_incident.get('start_at') or appearance.get('created_at') or '')
            matched_epoch = self._assistant_parse_datetime(matched_at, selected_zone)
            item = {'incident': matched_incident, 'event_id': representative_id, 'camera_id': str(matched_incident.get('camera_id') or appearance.get('camera_id') or ''), 'start_at': matched_at, 'seconds_from_anchor': round(matched_epoch.timestamp() - anchor_at.timestamp() if matched_epoch is not None else 0.0, 1), 'match_strength': 'appearance_similarity', 'confidence': round(similarity, 3), 'appearance_similarity': round(similarity, 4), 'reasons': [reason]}
            matches.append(item)
            matches_by_event_id[representative_id] = item
        strength_rank = {'confirmed_identity': 4, 'possible_identity': 3, 'appearance_similarity': 2, 'context_candidate': 1}
        matches = sorted(sorted(matches, key=lambda item: (-strength_rank.get(str(item.get('match_strength') or ''), 0), -float(item.get('confidence') or 0.0), abs(float(item.get('seconds_from_anchor') or 0.0))))[:min(call.limit, 12)], key=lambda item: str(item.get('start_at') or ''))
        confirmed = sum((item['match_strength'] == 'confirmed_identity' for item in matches))
        possible = sum((item['match_strength'] == 'possible_identity' for item in matches))
        contextual = sum((item['match_strength'] == 'context_candidate' for item in matches))
        appearance_similar = sum((item['match_strength'] == 'appearance_similarity' for item in matches))
        timeline_data = {'anchor_event_id': int(event_id) if event_id else None, 'anchor_camera_id': (anchor or {}).get('camera_id'), 'start_at': start.isoformat(), 'end_at': end.isoformat(), 'object_label': call.object_label, 'face_name': call.face_name, 'matches': [{key: item.get(key) for key in ('event_id', 'camera_id', 'start_at', 'seconds_from_anchor', 'match_strength', 'confidence', 'reasons', 'appearance_similarity')} for item in matches], 'limitations': ['Confirmed recognized faces can link incidents across cameras.', 'Possible face matches remain uncertain.', 'Shared person, vehicle, or animal labels plus nearby time provide context only.', 'Appearance similarity uses durable, model-versioned ReID vectors and is stronger than a shared class label, but it is not proof of identity.', 'Camera angle, lighting, occlusion, and visually similar subjects can change the score.', 'Only the strongest 12 candidates and at most 24 hours are returned.']}
        trace = AssistantEvidence(evidence_id=f'E-trace-{event_id or 'search'}', kind='cross_camera_timeline', title='Cross-camera investigation timeline', summary=f'Found {len(matches)} bounded timeline candidate(s): {confirmed} confirmed identity, {possible} possible identity, {appearance_similar} appearance-similar, and {contextual} context-only.', data=timeline_data, href=self._assistant_incident_evidence(anchor, int(event_id)).href if anchor and event_id else '/incidents', client_data={'timeline': timeline_data})
        evidence = [trace]
        if anchor and event_id:
            evidence.append(self._assistant_incident_evidence(anchor, int(event_id)))
        for item in matches:
            incident = item['incident']
            evidence.append(self._assistant_incident_evidence(incident, int(item['event_id'])))
        return evidence

    def _assistant_execute_tool(self, call: AssistantToolCall, request: AssistantChatRequest, active_config: AppConfig, active_manager: AppManager) -> list[AssistantEvidence]:
        if call.name == 'get_system_health':
            return [self._assistant_system_evidence(active_manager)]
        if call.name == 'get_camera_health':
            camera_id = call.camera_id or request.context.camera_id
            return self._assistant_camera_evidence(active_manager, camera_id)
        if call.name == 'explain_configuration':
            return [self._assistant_configuration_evidence(active_config)]
        if call.name == 'inspect_incident':
            event_id = call.event_id or request.context.incident_event_id
            item = self._assistant_inspect_incident(int(event_id), active_manager) if event_id else None
            return [item] if item is not None else []
        if call.name == 'analyze_incident_visual':
            event_id = call.event_id or request.context.incident_event_id
            if not event_id:
                return []
            if not self.deps.get_audit_ai_limiter().acquire(blocking=False):
                raise AuditAiError('another visual AI review is already running')
            try:
                item = self._assistant_visual_incident_evidence(int(event_id), active_config, active_manager)
            finally:
                self.deps.get_audit_ai_limiter().release()
            return [item] if item is not None else []
        if call.name == 'search_incidents':
            return self._assistant_search_incidents(call, request.context.time_zone, active_manager)
        if call.name == 'semantic_search_recordings':
            return self._assistant_semantic_search(call, request.context.time_zone, active_manager)
        if call.name == 'summarize_recent_activity':
            return [self._assistant_recent_activity_summary(call, request.context.time_zone, active_manager)]
        if call.name == 'trace_across_cameras':
            return self._assistant_trace_across_cameras(call, request, active_manager)
        if call.name == 'create_media_export':
            return [self._assistant_media_export_evidence(call, request, active_config)]
        return []

    def assistant_status(self) -> dict[str, Any]:
        ai = self.deps.get_config().audit_ai
        configured = bool(ai.enabled and ai.assistant_enabled and ai_provider_configured(ai))
        provider = AssistantProvider(ai)
        return {'enabled': bool(ai.assistant_enabled), 'configured': configured, 'provider': ai.provider, 'fast_model': provider.model_for_tier('fast'), 'reasoning_model': provider.model_for_tier('deep'), 'read_only': False, 'media_exports': True}

    async def assistant_chat(self, request: AssistantChatRequest) -> dict[str, Any]:
        with self.deps.manager_lock:
            active_config = self.deps.get_config()
            active_manager = self.deps.get_manager()
            ai = active_config.audit_ai
            if not ai.assistant_enabled:
                raise HTTPException(status_code=409, detail='SurvNG Assistant is disabled')
            if not ai.enabled:
                raise HTTPException(status_code=409, detail='AI features are disabled in Admin')
            if not ai_provider_configured(ai):
                raise HTTPException(status_code=409, detail='AI provider credentials are not configured')
            if not self.deps.get_assistant_limiter().acquire(blocking=False):
                raise HTTPException(status_code=429, detail='SurvNG Assistant is busy; try again shortly')
            self.deps.begin_ai_operation('assistant')

        def run() -> dict[str, Any]:
            provider = AssistantProvider(ai)
            catalog = self._assistant_catalog(active_config, active_manager)
            try:
                selected_zone = ZoneInfo(request.context.time_zone)
            except (ZoneInfoNotFoundError, ValueError):
                selected_zone = ZoneInfo('America/New_York')
            now = datetime.now(timezone.utc).astimezone(selected_zone)
            plan = provider.plan(request, catalog, now.isoformat())
            evidence: list[AssistantEvidence] = []
            seen: set[str] = set()
            for call in plan.tool_calls:
                for item in self._assistant_execute_tool(call, request, active_config, active_manager):
                    if item.evidence_id not in seen:
                        seen.add(item.evidence_id)
                        evidence.append(item)
            media_export = next((item for item in evidence if item.kind in {'media_export_job', 'media_export_clarification'}), None)
            answer = self._assistant_media_export_answer(media_export) if media_export is not None else provider.answer(request, evidence, plan.reasoning_tier)
            activity = next((item for item in evidence if item.kind == 'recent_activity_summary'), None)
            suggestions = self._assistant_activity_followups(activity) if activity is not None else answer.suggestions
            return {'message': answer.answer, 'citations': answer.citations, 'suggestions': suggestions, 'evidence': [item.client_payload() for item in evidence], 'tools': [call.name for call in plan.tool_calls], 'reasoning_tier': plan.reasoning_tier, 'model': provider.model_for_tier(plan.reasoning_tier), 'read_only': False}
        try:
            return await asyncio.to_thread(run)
        except AuditAiError as exc:
            LOGGER.warning('SurvNG Assistant provider failure: %s', exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception('SurvNG Assistant request failed')
            raise HTTPException(status_code=500, detail='SurvNG Assistant could not complete the request') from exc
        finally:
            self.deps.end_ai_operation('assistant')
            self.deps.get_assistant_limiter().release()

    def incident_ai_apply(self, event_id: int, request: IncidentAiApplyRequest) -> dict:
        if not request.confirmed:
            raise HTTPException(status_code=400, detail='explicit confirmation is required')
        if not request.changes:
            raise HTTPException(status_code=400, detail='no recommendation changes supplied')
        if any((change.scope != 'camera' for change in request.changes)):
            raise HTTPException(status_code=400, detail='incident reviews may only change camera-scoped motion settings')
        with self.deps.manager_lock:
            if not self.deps.get_config().audit_ai.allow_apply_recommendations:
                raise HTTPException(status_code=403, detail='applying AI recommendations is disabled')
            event = self.deps.get_manager().events.get(event_id)
            if event is None:
                raise HTTPException(status_code=404, detail='incident event not found')
            next_config = self.deps.get_config().model_copy(deep=True)
            camera = camera_by_id(next_config, str(event.get('camera_id') or ''))
            if camera is None:
                raise HTTPException(status_code=404, detail='incident camera not found')
            current_fingerprint = self._assistant_motion_config_fingerprint(next_config, camera)
            if request.configuration_fingerprint != current_fingerprint:
                raise HTTPException(status_code=409, detail='motion settings changed after this review; run visual analysis again')
            if not self._verify_ai_recommendation_token(request.recommendation_proof, kind='incident_visual', record_id=event_id, camera_id=camera.id, configuration_fingerprint=current_fingerprint, changes=request.changes):
                raise HTTPException(status_code=409, detail='AI recommendations are expired or do not match this visual review')
            try:
                changes, previews = self._assistant_motion_change_previews(next_config, camera, request.changes)
                if not changes:
                    raise ValueError('recommendations do not change active settings')
                for change in changes:
                    self._apply_pipeline_ai_change(next_config, camera, change, validate_tuning_value(change.setting, change.value))
                validate_motion_pipeline_configuration(next_config)
                next_config = AppConfig.model_validate(next_config.model_dump(mode='json'))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            _effective_config, apply_result = self.deps.apply_config_update(next_config)
        return {'ok': True, 'event_id': event_id, 'camera_id': camera.id, 'applied': previews, 'workers_restarted': bool(apply_result['camera_workers_restarted']), 'apply_mode': apply_result['apply_mode']}

def create_intelligence_router(deps: IntelligenceDependencies) -> IntelligenceRouteBundle:
    router = APIRouter()
    service = IntelligenceService(deps)
    router.add_api_route('/api/motion-audit', service.motion_audit, methods=['GET'])
    router.add_api_route('/api/motion-effectiveness', service.motion_effectiveness, methods=['GET'])
    router.add_api_route('/api/motion-audit/{audit_id}/snapshot.jpg', service.motion_audit_snapshot, methods=['GET'])
    router.add_api_route('/api/motion-audit/{audit_id}/ai-analyze', service.motion_audit_ai_analyze, methods=['POST'])
    router.add_api_route('/api/motion-audit/{audit_id}/ai-apply', service.motion_audit_ai_apply, methods=['POST'])
    router.add_api_route('/api/motion-ai-reviews', service.start_motion_ai_review, methods=['POST'])
    router.add_api_route('/api/motion-ai-reviews/latest', service.latest_motion_ai_review, methods=['GET'])
    router.add_api_route('/api/motion-ai-reviews/{review_id}', service.motion_ai_review, methods=['GET'])
    router.add_api_route('/api/motion-ai-reviews/{review_id}/apply', service.camera_intelligence_apply, methods=['POST'])
    router.add_api_route('/api/camera-intelligence/evaluations/latest', service.latest_camera_intelligence_evaluation, methods=['GET'])
    router.add_api_route('/api/camera-intelligence/evaluations/{evaluation_id}/follow-up', service.start_camera_intelligence_followup, methods=['POST'])
    router.add_api_route('/api/calibration/runs', service.start_calibration_run, methods=['POST'], status_code=202)
    router.add_api_route('/api/calibration/runs', service.calibration_runs, methods=['GET'])
    router.add_api_route('/api/calibration/runs/{run_id}', service.calibration_run, methods=['GET'])
    router.add_api_route('/api/calibration/runs/{run_id}/apply', service.calibration_apply, methods=['POST'])
    router.add_api_route('/api/calibration/change-sets', service.calibration_change_sets, methods=['GET'])
    router.add_api_route('/api/calibration/change-sets/{change_set_id}/evaluate', service.start_calibration_evaluation, methods=['POST'], status_code=202)
    router.add_api_route('/api/calibration/change-sets/{change_set_id}/rollback', service.calibration_rollback, methods=['POST'])
    router.add_api_route('/api/assistant/status', service.assistant_status, methods=['GET'])
    router.add_api_route('/api/assistant/chat', service.assistant_chat, methods=['POST'])
    router.add_api_route('/api/incidents/{event_id}/ai-apply', service.incident_ai_apply, methods=['POST'])
    handlers = {
        '_ai_recommendation_payload': service._ai_recommendation_payload,
        '_apply_pipeline_ai_change': service._apply_pipeline_ai_change,
        '_assistant_activity_followups': service._assistant_activity_followups,
        '_assistant_camera_evidence': service._assistant_camera_evidence,
        '_assistant_catalog': service._assistant_catalog,
        '_assistant_configuration_evidence': service._assistant_configuration_evidence,
        '_assistant_event_objects': service._assistant_event_objects,
        '_assistant_execute_tool': service._assistant_execute_tool,
        '_assistant_incident_evidence': service._assistant_incident_evidence,
        '_assistant_incident_payload': service._assistant_incident_payload,
        '_assistant_inspect_incident': service._assistant_inspect_incident,
        '_assistant_media_export_answer': service._assistant_media_export_answer,
        '_assistant_media_export_evidence': service._assistant_media_export_evidence,
        '_assistant_motion_change_current_value': service._assistant_motion_change_current_value,
        '_assistant_motion_change_previews': service._assistant_motion_change_previews,
        '_assistant_motion_config_fingerprint': service._assistant_motion_config_fingerprint,
        '_assistant_parse_datetime': service._assistant_parse_datetime,
        '_assistant_prioritize_trace_candidates': service._assistant_prioritize_trace_candidates,
        '_assistant_recent_activity_summary': service._assistant_recent_activity_summary,
        '_assistant_search_incidents': service._assistant_search_incidents,
        '_assistant_semantic_search': service._assistant_semantic_search,
        '_assistant_system_evidence': service._assistant_system_evidence,
        '_assistant_trace_across_cameras': service._assistant_trace_across_cameras,
        '_assistant_visual_incident_evidence': service._assistant_visual_incident_evidence,
        '_audit_ai_context': service._audit_ai_context,
        '_calibration_camera_review': service._calibration_camera_review,
        '_calibration_followup_monitor': service._calibration_followup_monitor,
        '_camera_intelligence_candidates': service._camera_intelligence_candidates,
        '_issue_ai_recommendation_token': service._issue_ai_recommendation_token,
        '_run_calibration_evaluation': service._run_calibration_evaluation,
        '_run_camera_intelligence_review': service._run_camera_intelligence_review,
        '_run_motion_ai_review': service._run_motion_ai_review,
        '_run_system_calibration': service._run_system_calibration,
        '_verify_ai_recommendation_token': service._verify_ai_recommendation_token,
        'assistant_chat': service.assistant_chat,
        'assistant_status': service.assistant_status,
        'calibration_apply': service.calibration_apply,
        'calibration_change_sets': service.calibration_change_sets,
        'calibration_rollback': service.calibration_rollback,
        'calibration_run': service.calibration_run,
        'calibration_runs': service.calibration_runs,
        'camera_intelligence_apply': service.camera_intelligence_apply,
        'incident_ai_apply': service.incident_ai_apply,
        'latest_camera_intelligence_evaluation': service.latest_camera_intelligence_evaluation,
        'latest_motion_ai_review': service.latest_motion_ai_review,
        'motion_ai_review': service.motion_ai_review,
        'motion_audit': service.motion_audit,
        'motion_audit_ai_analyze': service.motion_audit_ai_analyze,
        'motion_audit_ai_apply': service.motion_audit_ai_apply,
        'motion_audit_snapshot': service.motion_audit_snapshot,
        'motion_effectiveness': service.motion_effectiveness,
        'start_calibration_evaluation': service.start_calibration_evaluation,
        'start_calibration_run': service.start_calibration_run,
        'start_camera_intelligence_followup': service.start_camera_intelligence_followup,
        'start_motion_ai_review': service.start_motion_ai_review,
    }
    return IntelligenceRouteBundle(router=router, service=service, handlers=handlers)
