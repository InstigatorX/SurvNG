from pathlib import Path
import ast
import textwrap


def replace(text, old, new, count=1):
    assert text.count(old) == count, f'anchor count {text.count(old)} != {count}: {old[:100]!r}'
    return text.replace(old, new)


def method(text, name, replacement):
    tree = ast.parse(text)
    node = next(n for c in tree.body if isinstance(c, ast.ClassDef)
                for n in c.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    lines = text.splitlines(keepends=True)
    lines[node.lineno-1:node.end_lineno] = [textwrap.indent(textwrap.dedent(replacement).strip(), '    ') + '\n']
    return ''.join(lines)


path = Path('survng/app/native_evidence.py')
s = path.read_text()
s = replace(s, 'from collections import Counter', 'from collections import Counter, deque')
s = replace(s, '        self.counts = Counter()\n', '        self.counts = Counter()\n        self.recent = deque(maxlen=32)\n')
s = replace(s, 'return {"queued": len(self._pending), "active": len(self._active), "counters": dict(self.counts), "admission": self.admission.status()}',
'''return {"queued": sum(job["due"] != float("inf") for job in self._pending.values()),
                    "retained": sum(job["due"] == float("inf") for job in self._pending.values()),
                    "active": len(self._active), "counters": dict(self.counts),
                    "recent": list(self.recent), "admission": self.admission.status()}''')
s = replace(s, '            job["candidates"] = shortlist([*job["candidates"], candidate])\n',
'''            job["candidates"] = shortlist([*job["candidates"], candidate])
            job["version"] = job.get("version", 0) + 1
            job["due"] = min(job["due"], time.monotonic()+20)
            if not job.get("recorded_history"):
                job["deadline"] = time.monotonic()+120
''')
s = replace(s, '            job["recorded_history"] = True\n',
'''            job["recorded_history"] = True
            job["version"] = job.get("version", 0) + 1
            job["due"] = min(job["due"], time.monotonic()+15)
''')
s = method(s, '_run', '''
def _run(self):
    while True:
        with self._condition:
            if self._closed:
                return
            now = time.monotonic()
            for key, cached in list(self._pending.items()):
                if key not in self._active and cached["due"] == float("inf") and cached["deadline"] <= now:
                    del self._pending[key]
                    self.counts["retained_expired"] += 1
            due = [(key, job) for key, job in self._pending.items()
                   if key not in self._active and job["due"] <= now]
            if not due:
                self._condition.wait(1)
                continue
            event_id, job = min(due, key=lambda item: item[1]["due"])
            # Keep a single job owner while it runs. Completion/new frames may
            # update it concurrently, but cannot discard the retained shortlist.
            version = job.get("version", 0)
            candidates = list(job["candidates"])
            recorded_history = bool(job.get("recorded_history"))
            job["due"] = float("inf")
            self._active.add(event_id)
        try:
            result = self.process(event_id, candidates, recorded_history=recorded_history)
        except Exception as exc:
            result = {"event_id": event_id, "status": "failed", "reason": "processing_error",
                      "error_type": type(exc).__name__}
            if not self._closed:
                LOGGER.exception("Native evidence failed for event %s", event_id)
                self._record_result(result)
        finally:
            with self._condition:
                self._active.discard(event_id)
        self.counts[result["status"]] += 1
        exhausted = False
        with self._condition:
            if self._closed or self._pending.get(event_id) is not job:
                continue
            changed = job.get("version", 0) != version
            retry = result["status"] in {"recording_pending", "failed"}
            if retry and time.monotonic() < job["deadline"]:
                job["due"] = min(job["due"], time.monotonic()+10)
            elif changed:
                # enqueue/offer already set the next due time and kept the
                # terminal rescan flag. Never overwrite newer work with a retry.
                continue
            elif recorded_history or result["status"] in {"event_missing", "not_native"}:
                self._pending.pop(event_id)
                exhausted = retry
            else:
                # A preview may succeed before completion. Retain its three
                # exact live images, bounded by queue capacity and idle expiry.
                job["due"] = float("inf")
                exhausted = retry
        if exhausted:
            self._record_result(dict(result, retry_exhausted=True))
''')
s = replace(s, '    def recorded_candidates(self, event, tracking):',
            '    def recorded_candidates(self, event, tracking, retained=()):')
s = replace(s, '        for epoch in epochs:\n            if self._closed:',
'''        for epoch in epochs:
            if any(abs(epoch - candidate.epoch) < 0.5 for candidate in retained):
                continue
            if self._closed:''')
s = method(s, 'process', '''
def _record_result(self, result):
    summary = {"source": "native_cover", "recorded_at_epoch": time.time(), **result}
    try:
        if result["status"] not in {"event_missing", "not_native"} and self.events.get(result["event_id"]):
            self.events.record_evidence_attempt(result["event_id"], summary)
    except Exception as exc:
        self.counts["diagnostics_failed"] += 1
        LOGGER.warning("Native cover diagnostics unavailable for event %s (%s)",
                       result["event_id"], type(exc).__name__)
    with self._condition:
        self.recent.append(summary)


def process(self, event_id, candidates=None, *, recorded_history=False):
    assets = []
    try:
        result = self._process(event_id, candidates, assets, recorded_history=recorded_history)
        self._record_result(result)
        return result
    finally:
        # Durable references protect every successfully archived image.
        for path, _ in assets:
            self.events._delete_snapshot_if_unreferenced(path, preserve_archive=True)
''')
s = replace(s, '    def _process(self, event_id, candidates, assets):',
            '    def _process(self, event_id, candidates, assets, *, recorded_history=False):')
s = replace(s, '''        pending = False
        if not candidates:
            candidates, pending = self.recorded_candidates(event, tracking)
        best = None''', '''        pending = False
        retained = list(candidates or [])
        if recorded_history or not retained:
            recorded, pending = self.recorded_candidates(event, tracking, retained)
            candidates = [*retained, *recorded]
        else:
            candidates = retained
        # Cover selection is bounded and independent of calibration history.
        candidates = shortlist(candidates)
        details = {"retained_live_candidates": len(retained),
                   "cover_candidates": len(candidates), "reasons": {}}
        failures = Counter()
        if not candidates:
            return {"event_id": event_id, "status": "recording_pending" if pending else "no_usable_candidate",
                    "reason": "live_recording_unavailable" if pending else "no_cover_candidate", **details}
        best = None''')
s = replace(s, '''        if calibrate:
            try:''', '''        epochs = calibration_epochs(tracking) if calibrate else []
        details["calibration"] = "verified" if same_fov_aligned else "unverified"
        if calibrate and (len(epochs) < 5 or epochs[-1] - epochs[0] < 3):
            details["calibration"] = "insufficient_history"
            calibrate = False
        if calibrate:
            try:''')
s = replace(s, '''                if require_verification:
                    raise
                self.counts["calibration_unavailable"] += 1''', '''                details["calibration"] = "unavailable"
                self.counts["calibration_unavailable"] += 1''')
s = replace(s, 'return self._process(event_id, candidates, assets)',
            'return self._process(event_id, candidates, assets, recorded_history=False)')
start = s.index('        # Without main/live clock alignment, projected same-FOV boxes can be')
end = s.index('            score = candidate_score(main, objects)', start)
old = s[start:end]
# Preserve the existing aligned/non-same-FOV branch rather than reformat it.
branch = old[old.index('            main_epoch = ('):]
new = '''        frame_verification = same_fov and not same_fov_aligned
        for candidate in candidates:
            if self._closed:
                break
            if frame_verification:
                # A still-image check is not recording-clock calibration. Use
                # only a box detected on the returned main image, never a
                # projected box or another timestamp's annotation.
                self.verifier.config = self.config.detector
                try:
                    checked = self.admission.verify(event["camera_id"], [candidate])
                except Exception as exc:
                    failures["main_verification_unavailable"] += 1
                    pending = True
                    LOGGER.warning("Native cover verification unavailable for event %s (%s)",
                                   event_id, type(exc).__name__)
                    break
                cover = checked.get("cover") if checked.get("status") == "confirmed" else None
                if cover is None:
                    votes = [vote for check in checked.get("checks", []) for vote in check.get("votes", [])]
                    if "unavailable" in votes or "unavailable" in checked.get("votes", []):
                        pending = True
                        failures["main_recording_unavailable"] += 1
                    else:
                        failures["main_verification_failed"] += 1
                    continue
                main, detected, main_epoch = cover
                objects = [deepcopy(detected)]
                objects[0].pop("mask_polygon", None)
                objects[0].update(
                    frame_source="recorded_main", frame_captured_at_epoch=main_epoch,
                    detection_frame_width=main.shape[1], detection_frame_height=main.shape[0],
                    snapshot_visible=True, native_cover_verified=True,
                    box_provenance="detected_in_main",
                    verification={"status": "confirmed", "source": "main"},
                    native_alignment={"method": "main_frame_verified", "nomination_epoch": candidate.epoch,
                                      "sample_offset_seconds": main_epoch - candidate.epoch},
                )
            else:
'''
new += textwrap.indent(branch, '    ')
s = s[:start] + new + s[end:]
s = replace(s, '''                self.counts["main_rejected"] += 1
                continue''', '''                self.counts["main_rejected"] += 1
                failures["image_quality_rejected"] += 1
                continue''')
s = replace(s, '''            if main.shape[0] * main.shape[1] <= candidate.image.shape[0] * candidate.image.shape[1]:
                continue''', '''            if main.shape[0] * main.shape[1] <= candidate.image.shape[0] * candidate.image.shape[1]:
                failures["main_not_higher_resolution"] += 1
                continue''')
s = replace(s, '''            if output_path is None:
                continue''', '''            if output_path is None:
                failures["image_write_failed"] += 1
                pending = True
                continue''')
s = replace(s, '''        if not best:
            return {''', '''        details["reasons"] = dict(failures)
        if not best:
            reason = next((name for name in ("main_verification_unavailable", "image_write_failed",
                                           "main_recording_unavailable", "main_verification_failed",
                                           "image_quality_rejected", "main_not_higher_resolution")
                           if failures[name]), "no_usable_candidate")
            return {
                **details,
                "reason": reason,''')
s = replace(s, '''        result = self.events.promote_native_evidence(
            event_id, output_path, objects, assets, score
        )''', '''        adoption = {}
        result = self.events.promote_native_evidence(
            event_id, output_path, objects, assets, score, diagnostics=adoption
        )''')
s = replace(s, '''            "status": "promoted" if result else "kept_better_cover",
            "width": width,''', '''            "status": ("promoted" if result else "kept_better_cover"
                       if adoption.get("reason") == "better_cover_retained" else "recording_pending"),
            "reason": (objects[0].get("native_alignment", {}).get("method", "verified_main")
                       if result else adoption.get("reason", "cover_not_adopted")),
            **details,
            "width": width,''')
# Give decode misses on the aligned path a distinct operational reason as well.
s = replace(s, '''                if main is None:
                    pending = True
                    continue
                objects = (''', '''                if main is None:
                    pending = True
                    failures["main_recording_unavailable"] += 1
                    continue
                objects = (''')
ast.parse(s)
path.write_text(s)

path = Path('survng/app/event_store/store.py')
s = path.read_text()
s = replace(s, '    def promote_native_evidence(self, event_id, snapshot_path, objects, assets, score):',
            '    def promote_native_evidence(self, event_id, snapshot_path, objects, assets, score, *, diagnostics=None):')
start=s.index('    def promote_native_evidence(')
end=s.index('    def update_native_replay_alignment(', start)
part=s[start:end]
part=replace(part, '        stale = []\n', '        stale = []\n        if diagnostics is not None:\n            diagnostics["reason"] = "event_missing"\n')
part=replace(part, '            if row is not None and not snapshot_deletion_claimed',
'''            if row is not None and diagnostics is not None:
                diagnostics["reason"] = "snapshot_deletion_pending"
            if row is not None and not snapshot_deletion_claimed''')
part=replace(part, '                if score > previous_score + 0.05:',
'''                if diagnostics is not None:
                    diagnostics["reason"] = "better_cover_retained"
                if score > previous_score + 0.05:''')
part=replace(part, '                    updated = self._finish_evidence_commit',
'''                    if diagnostics is not None:
                        diagnostics["reason"] = "promoted"
                    updated = self._finish_evidence_commit''')
s=s[:start]+part+s[end:]
ast.parse(s); path.write_text(s)

path=Path('survng/app/native_routes.py')
s=path.read_text()
s=replace(s, '    return router\n', '''    @router.get("/api/events/{event_id}/native-evidence")
    def native_evidence(event_id: int):
        events = get_manager().events
        event = events.get(event_id)
        if event is None:
            raise HTTPException(404, "event not found")
        return {"event_id": event_id, "camera_id": event["camera_id"],
                "attempts": [attempt for attempt in events.evidence_attempts(event_id)
                             if attempt.get("source") == "native_cover"]}

    return router
''')
ast.parse(s); path.write_text(s)

path=Path('tests/test_native_evidence.py')
s=path.read_text()
s=replace(s, '''    def process(eid, candidates):
        assert eid == event_id
        assert candidates is None''', '''    def process(eid, candidates, *, recorded_history=False):
        assert eid == event_id
        assert recorded_history
        assert [candidate.epoch for candidate in candidates] == [101]''')
start=s.index('def test_optional_calibration_failure_keeps_substream_when_timing_unverified(')
end=s.index('\ndef test_verification_projection',start)
part=s[start:end]
part=replace(part, "assert result['status'] == 'no_usable_candidate'", "assert result['status'] == 'recording_pending'\n    assert result['reason'] == 'main_verification_unavailable'")
part=replace(part, "assert service.counts['calibration_unavailable'] == 1", "assert result['calibration'] == 'insufficient_history'")
s=s[:start]+part+s[end:]
start=s.index('def test_same_fov_unverified_timing_never_promotes_projected_main_without_verification(')
part=s[start:]
part=replace(part, '''    service.verifier.detect = Mock(return_value=[
        dict(obj, box={key: value * 2 for key, value in obj['box'].items()})
    ])''', '''    service.verifier.detect = Mock(return_value=[])''')
part=replace(part, "assert service.counts['replay_alignment_unverified'] == 1", "assert result['reason'] == 'main_verification_failed'")
s=s[:start]+part
ast.parse(s); path.write_text(s)
