"""Build high-value ReID review candidates for environment adaptation."""

from __future__ import annotations

from typing import Any, Callable

from .store import ReidTrainingStore


class ReidTrainingReviewService:
    """Surface cross-camera hard pairs and unreviewed track galleries."""

    def __init__(
        self,
        store: ReidTrainingStore,
        appearance_matches: Callable[..., list[dict[str, Any]]],
    ) -> None:
        self.store = store
        self.appearance_matches = appearance_matches

    def review_queue(
        self,
        *,
        limit: int = 20,
        hours: float = 168.0,
        event_scan_limit: int = 40,
    ) -> dict[str, Any]:
        bounded = max(1, min(int(limit), 100))
        hard_pairs = self._hard_pairs(
            limit=bounded,
            hours=max(1.0, float(hours)),
            event_scan_limit=max(1, min(int(event_scan_limit), 200)),
        )
        remaining = max(0, bounded - len(hard_pairs))
        tracks = self._unreviewed_tracks(limit=remaining) if remaining else []
        return {
            "hard_pairs": hard_pairs,
            "tracks": tracks,
            "status": self.store.status(),
        }

    def apply_review(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = str(payload.get("action") or "").strip().lower()
        if action == "confirm_same":
            return self._confirm_same(payload)
        if action == "mark_different":
            return self._mark_different(payload)
        if action == "unknown":
            return self._unknown_pair(payload)
        if action == "reject":
            return self._reject(payload)
        raise ValueError(f"unsupported ReID review action: {action}")

    def _hard_pairs(
        self,
        *,
        limit: int,
        hours: float,
        event_scan_limit: int,
    ) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        seen: set[tuple[int, int, int, int]] = set()
        for event_id in self.store.recent_event_ids(limit=event_scan_limit):
            if len(pairs) >= limit:
                break
            try:
                matches = self.appearance_matches(
                    event_id,
                    hours=hours,
                    limit=24,
                    cross_camera_only=True,
                )
            except Exception:
                continue
            for match in matches:
                if not isinstance(match, dict):
                    continue
                if str(match.get("anchor_label") or "").lower() != "person":
                    continue
                if str(match.get("candidate_label") or "").lower() != "person":
                    continue
                try:
                    left_event = int(event_id)
                    left_track = int(match.get("anchor_track_id"))
                    right_event = int(match["event_id"])
                    right_track = int(match["candidate_track_id"])
                    similarity = float(match.get("similarity") or 0.0)
                    threshold = float(match.get("threshold") or 0.0)
                except (KeyError, TypeError, ValueError):
                    continue
                key = tuple(sorted(((left_event, left_track), (right_event, right_track))))
                flat_key = (key[0][0], key[0][1], key[1][0], key[1][1])
                if flat_key in seen:
                    continue
                if self.store.pair_reviewed(left_event, left_track, right_event, right_track):
                    continue
                left_samples = self.store.samples_for_track(left_event, left_track)
                right_samples = self.store.samples_for_track(right_event, right_track)
                if not left_samples or not right_samples:
                    continue
                left_person = left_samples[0].get("assigned_person_id")
                right_person = right_samples[0].get("assigned_person_id")
                if (
                    left_person is not None
                    and right_person is not None
                    and int(left_person) == int(right_person)
                ):
                    # Already same identity from a prior merge; skip.
                    continue
                seen.add(flat_key)
                margin = abs(similarity - threshold)
                pairs.append({
                    "kind": "hard_pair",
                    "priority": round(
                        (1.0 if bool(match.get("visually_similar")) else 0.6)
                        + max(0.0, 0.25 - margin),
                        4,
                    ),
                    "similarity": round(similarity, 4),
                    "threshold": round(threshold, 4),
                    "visually_similar": bool(match.get("visually_similar")),
                    "left": {
                        "event_id": left_event,
                        "track_id": left_track,
                        "camera_id": str(left_samples[0]["camera_id"]),
                        "person_id": left_person,
                        "samples": left_samples[:4],
                    },
                    "right": {
                        "event_id": right_event,
                        "track_id": right_track,
                        "camera_id": str(right_samples[0]["camera_id"]),
                        "person_id": right_person,
                        "samples": right_samples[:4],
                    },
                })
                if len(pairs) >= limit:
                    break
        pairs.sort(key=lambda item: (-float(item["priority"]), -float(item["similarity"])))
        return pairs[:limit]

    def _unreviewed_tracks(self, *, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        samples = self.store.list_samples(limit=max(limit * 8, 40), review_status="auto")
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for sample in samples:
            key = (int(sample["event_id"]), int(sample["track_id"]))
            grouped.setdefault(key, []).append(sample)
        tracks: list[dict[str, Any]] = []
        for (event_id, track_id), items in grouped.items():
            tracks.append({
                "kind": "track",
                "event_id": event_id,
                "track_id": track_id,
                "camera_id": str(items[0]["camera_id"]),
                "person_id": items[0].get("assigned_person_id"),
                "sample_count": len(items),
                "samples": items[:6],
            })
            if len(tracks) >= limit:
                break
        return tracks

    def _confirm_same(self, payload: dict[str, Any]) -> dict[str, Any]:
        left, right = self._pair_sides(payload)
        left_person = left["samples"][0].get("assigned_person_id")
        right_person = right["samples"][0].get("assigned_person_id")
        if left_person is None and right_person is None:
            keep = self.store.create_identity()
            self.store.assign_track_identity(left["event_id"], left["track_id"], keep)
            self.store.assign_track_identity(right["event_id"], right["track_id"], keep)
            absorb = None
        elif left_person is None:
            keep = int(right_person)
            absorb = None
            self.store.assign_track_identity(left["event_id"], left["track_id"], keep)
        elif right_person is None:
            keep = int(left_person)
            absorb = None
            self.store.assign_track_identity(right["event_id"], right["track_id"], keep)
        else:
            keep = int(left_person)
            absorb = int(right_person)
            if keep != absorb:
                self.store.merge_track_identities(
                    keep_person_id=keep,
                    absorb_person_id=absorb,
                )
            self.store.assign_track_identity(left["event_id"], left["track_id"], keep)
            self.store.assign_track_identity(right["event_id"], right["track_id"], keep)
        self.store.record_pair_review(
            left_event_id=left["event_id"],
            left_track_id=left["track_id"],
            right_event_id=right["event_id"],
            right_track_id=right["track_id"],
            decision="same",
            similarity=payload.get("similarity"),
            left_sample_id=str(left["samples"][0]["sample_id"]),
            right_sample_id=str(right["samples"][0]["sample_id"]),
        )
        return {
            "action": "confirm_same",
            "person_id": keep,
            "absorbed_person_id": absorb,
        }

    def _mark_different(self, payload: dict[str, Any]) -> dict[str, Any]:
        left, right = self._pair_sides(payload)
        left_person = left["samples"][0].get("assigned_person_id")
        right_person = right["samples"][0].get("assigned_person_id")
        if left_person is None:
            left_person = self.store.create_identity()
            self.store.assign_track_identity(
                left["event_id"],
                left["track_id"],
                int(left_person),
            )
        if right_person is None or int(right_person) == int(left_person):
            right_person = self.store.create_identity()
            self.store.assign_track_identity(
                right["event_id"],
                right["track_id"],
                int(right_person),
            )
        else:
            self.store.assign_track_identity(
                right["event_id"],
                right["track_id"],
                int(right_person),
            )
            self.store.assign_track_identity(
                left["event_id"],
                left["track_id"],
                int(left_person),
            )
        self.store.record_pair_review(
            left_event_id=left["event_id"],
            left_track_id=left["track_id"],
            right_event_id=right["event_id"],
            right_track_id=right["track_id"],
            decision="different",
            similarity=payload.get("similarity"),
            left_sample_id=str(left["samples"][0]["sample_id"]),
            right_sample_id=str(right["samples"][0]["sample_id"]),
        )
        return {
            "action": "mark_different",
            "left_person_id": int(left_person),
            "right_person_id": int(right_person),
        }

    def _unknown_pair(self, payload: dict[str, Any]) -> dict[str, Any]:
        left, right = self._pair_sides(payload)
        self.store.record_pair_review(
            left_event_id=left["event_id"],
            left_track_id=left["track_id"],
            right_event_id=right["event_id"],
            right_track_id=right["track_id"],
            decision="unknown",
            similarity=payload.get("similarity"),
            left_sample_id=str(left["samples"][0]["sample_id"]),
            right_sample_id=str(right["samples"][0]["sample_id"]),
        )
        return {"action": "unknown"}

    def _reject(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample_id = str(payload.get("sample_id") or "").strip()
        if sample_id:
            if not self.store.set_sample_review_status(
                sample_id,
                "rejected",
                assignment_source="manual",
            ):
                raise ValueError("sample not found")
            return {"action": "reject", "sample_id": sample_id}
        left, right = self._pair_sides(payload)
        side = str(payload.get("side") or "both").strip().lower()
        updated = 0
        if side in {"left", "both"}:
            updated += self.store.reject_track(left["event_id"], left["track_id"])
        if side in {"right", "both"}:
            updated += self.store.reject_track(right["event_id"], right["track_id"])
        self.store.record_pair_review(
            left_event_id=left["event_id"],
            left_track_id=left["track_id"],
            right_event_id=right["event_id"],
            right_track_id=right["track_id"],
            decision="reject",
            similarity=payload.get("similarity"),
            left_sample_id=str(left["samples"][0]["sample_id"]),
            right_sample_id=str(right["samples"][0]["sample_id"]),
        )
        return {"action": "reject", "updated_samples": updated}

    def _pair_sides(
        self,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            left_event = int(payload["left_event_id"])
            left_track = int(payload["left_track_id"])
            right_event = int(payload["right_event_id"])
            right_track = int(payload["right_track_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("pair review requires left/right event_id and track_id") from exc
        left_samples = self.store.samples_for_track(left_event, left_track)
        right_samples = self.store.samples_for_track(right_event, right_track)
        if not left_samples or not right_samples:
            raise ValueError("both tracks must have training samples")
        return (
            {
                "event_id": left_event,
                "track_id": left_track,
                "samples": left_samples,
            },
            {
                "event_id": right_event,
                "track_id": right_track,
                "samples": right_samples,
            },
        )
