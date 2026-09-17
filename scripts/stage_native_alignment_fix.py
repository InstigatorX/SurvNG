from pathlib import Path


def strip_indent(text: str) -> str:
    return text.replace("          ", "")


path = Path("survng/app/native_evidence.py")
text = path.read_text()

anchor = strip_indent('''def shortlist(candidates, limit=3):
          selected = []
          for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
              if all(abs(candidate.epoch - previous.epoch) >= 1 for previous in selected):
                  selected.append(candidate)
              if len(selected) == limit:
                  break
          return selected


      class NativeEvidenceService:
''')
replacement = strip_indent('''def shortlist(candidates, limit=3):
          selected = []
          for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
              if all(abs(candidate.epoch - previous.epoch) >= 1 for previous in selected):
                  selected.append(candidate)
              if len(selected) == limit:
                  break
          return selected


      def calibration_epochs(tracking, limit=9):
          """Select bounded track-history timestamps independently of cover images."""
          epochs = []
          for track in tracking.get("tracks") or []:
              for sample in track.get("box_history") or []:
                  if len(sample) < 5:
                      continue
                  values = sample[:5]
                  if not all(isinstance(value, (int, float)) and np.isfinite(value) for value in values):
                      continue
                  epoch, x1, y1, x2, y2 = values
                  if x2 <= x1 or y2 <= y1:
                      continue
                  epochs.append(float(epoch))
          ordered = sorted(set(epochs))
          if len(ordered) <= limit:
              return ordered
          indices = sorted(set(int(round(index)) for index in np.linspace(0, len(ordered) - 1, limit)))
          return [ordered[index] for index in indices]


      class NativeEvidenceService:
''')
if anchor not in text:
    raise SystemExit("shortlist anchor not found")
text = text.replace(anchor, replacement, 1)

method_anchor = "    def project_main(self, candidate, main):\n"
calibration_method = strip_indent('''    def replay_calibration_observations(self, event, tracking):
          """Detect on bounded main-stream timestamps chosen from full track history."""
          width = tracking.get("frame_width", 0)
          height = tracking.get("frame_height", 0)
          if not width or not height:
              return [], False
          observations = []
          pending = False
          self.verifier.config = self.config.detector
          for epoch in calibration_epochs(tracking):
              if self._closed:
                  break
              main = self.read_frame(event["camera_id"], epoch, "main")
              if main is None:
                  pending = True
                  continue
              detections = self.verifier.detect(main)
              observations.append({
                  "epoch": epoch,
                  "objects": resize_objects(
                      detections,
                      (main.shape[1], main.shape[0]),
                      (width, height),
                  ),
              })
          return observations, pending

''')
if method_anchor not in text:
    raise SystemExit("project_main anchor not found")
text = text.replace(method_anchor, calibration_method + method_anchor, 1)

start = text.index("    def _process(self, event_id, candidates, assets):\n")
new_process = strip_indent('''    def _process(self, event_id, candidates, assets):
          event = self.events.get(event_id)
          if not event:
              return {"event_id": event_id, "status": "event_missing"}
          _, tracking = event_tracking(event)
          if tracking.get("implementation") != "gvatrack":
              return {"event_id": event_id, "status": "not_native"}
          pending = False
          if not candidates:
              candidates, pending = self.recorded_candidates(event, tracking)
          best = None
          camera = next((camera for camera in self.config.cameras if camera.id == event["camera_id"]), None)
          same_fov = bool(camera and camera.native_same_field_of_view)
          recording_alignment = tracking.get("recording_alignment") or {}
          replay_offset = recording_alignment.get("offset_seconds")
          same_fov_aligned = bool(
              same_fov
              and recording_alignment.get("source") == "main"
              and recording_alignment.get("verified") is True
              and isinstance(replay_offset, (int, float))
              and -3 < float(replay_offset) < 3
          )
          require_verification = self.config.detector.native.verification_enabled
          calibrate = bool(
              same_fov
              and not same_fov_aligned
              and tracking.get("state") == "complete"
              and tracking.get("native_session")
              and tracking.get("frame_width")
              and tracking.get("frame_height")
          )

          # Main and live streams may have independent recording clocks. Cover
          # selection keeps only three live images, which is intentionally too
          # small for trajectory calibration. Calibrate from the full persisted
          # track history instead, then rerun cover promotion at the corrected
          # main-stream timestamp.
          if calibrate:
              try:
                  timing_observations, calibration_pending = self.replay_calibration_observations(
                      event, tracking
                  )
              except Exception as exc:
                  if require_verification:
                      raise
                  self.counts["calibration_unavailable"] += 1
                  LOGGER.warning(
                      "Native replay calibration unavailable for event %s (%s)",
                      event_id,
                      type(exc).__name__,
                  )
              else:
                  pending = pending or calibration_pending
                  alignment = estimate_replay_alignment(tracking, timing_observations)
                  if alignment:
                      aligned = self.events.update_native_replay_alignment(
                          event_id, tracking.get("native_session"), alignment
                      )
                      if aligned:
                          self.counts["replay_aligned"] += 1
                          self.publish(
                              "incident_update",
                              {
                                  "camera_id": event["camera_id"],
                                  "event_id": event_id,
                                  "evidence_revision": aligned["evidence_revision"],
                              },
                          )
                          return self._process(event_id, candidates, assets)

          # Without main/live clock alignment, projected same-FOV boxes can be
          # geometrically perfect but temporally wrong. When main verification
          # is disabled, retain the existing correctly annotated live/substream
          # evidence rather than publish a confident-looking misplaced box.
          if same_fov and not same_fov_aligned and not require_verification:
              waiting = pending or tracking.get("state") != "complete"
              self.counts[
                  "replay_alignment_pending" if waiting else "replay_alignment_unverified"
              ] += 1
              return {
                  "event_id": event_id,
                  "status": "recording_pending" if waiting else "no_usable_candidate",
              }

          for candidate in candidates[:12]:
              if self._closed:
                  break
              main_epoch = (
                  candidate.epoch - float(replay_offset)
                  if same_fov_aligned
                  else candidate.epoch
              )
              main = self.read_frame(event["camera_id"], main_epoch, "main")
              if main is None:
                  pending = True
                  continue
              objects = (
                  self.same_fov_main(candidate, main, main_epoch, float(replay_offset))
                  if same_fov_aligned
                  else self.match_main(candidate, main)
              )
              self.verifier.config = self.config.detector
              detections = []
              if require_verification and objects:
                  detections = self.verifier.detect(main)
              if require_verification:
                  verified = []
                  for obj in objects:
                      expected = obj["box"]
                      matching = []
                      for detected in detections:
                          threshold = self.config.detector.event_class_confidence_thresholds.get(
                              obj["label"], self.config.detector.confidence_threshold
                          )
                          if (
                              detected.get("label") != obj["label"]
                              or detected.get("confidence", 0) < threshold
                          ):
                              continue
                          actual = detected.get("box") or {}
                          if (
                              all(k in actual for k in ("x1", "y1", "x2", "y2"))
                              and matches_object_extent(expected, actual)
                          ):
                              matching.append(detected)
                      if matching:
                          actual = max(matching, key=lambda x: x.get("confidence", 0))
                          obj.update(
                              box=actual["box"],
                              confidence=actual["confidence"],
                              native_cover_verified=True,
                              box_provenance="detected_in_main",
                              verification={"status": "confirmed", "source": "main"},
                          )
                          verified.append(obj)
                  objects = verified
              else:
                  for obj in objects:
                      obj.update(
                          native_cover_verified=False,
                          box_provenance="projected_from_substream",
                          verification={"status": "confirmed", "source": "substream"},
                      )
              score = candidate_score(main, objects)
              if score is None:
                  self.counts["main_rejected"] += 1
                  continue
              if main.shape[0] * main.shape[1] <= candidate.image.shape[0] * candidate.image.shape[1]:
                  continue
              for obj in objects:
                  obj["native_cover_score"] = score
                  obj["snapshot_quality_score"] = min(1.0, score / 6)
                  box = obj["box"]
                  obj["snapshot_subject_area_ratio"] = (
                      (box["x2"] - box["x1"])
                      * (box["y2"] - box["y1"])
                      / (main.shape[0] * main.shape[1])
                  )
                  obj["snapshot_edge_clearance_ratio"] = min(
                      box["x1"] / main.shape[1],
                      box["y1"] / main.shape[0],
                      1 - box["x2"] / main.shape[1],
                      1 - box["y2"] / main.shape[0],
                  )
                  obj["snapshot_primary_subject"] = True
                  obj["temporal_sample_offset_seconds"] = (
                      candidate.epoch - datetime.fromisoformat(event["created_at"]).timestamp()
                  )
              if len(assets) >= 3 and score <= min(
                  items[1][0]["native_cover_score"] for items in assets
              ):
                  continue
              directory = self.media_storage.directory(
                  "snapshots", event["camera_id"], event["camera_id"]
              )
              path = self.image_writer.write(
                  directory, f"native-{event_id}-{uuid.uuid4().hex}", main
              )
              if path is None:
                  continue
              assets.append((str(path), objects))
              if len(assets) > 3:
                  lowest = min(assets, key=lambda item: item[1][0]["native_cover_score"])
                  assets.remove(lowest)
                  Path(lowest[0]).unlink(missing_ok=True)
              if best is None or score > best[0]:
                  best = (score, str(path), objects, main.shape[1], main.shape[0])
          if not best:
              return {
                  "event_id": event_id,
                  "status": (
                      "recording_pending"
                      if pending
                      else "no_verified_candidate"
                      if require_verification
                      else "no_usable_candidate"
                  ),
              }
          score, path, objects, width, height = best
          result = self.events.promote_native_evidence(event_id, path, objects, assets, score)
          if result:
              self.publish(
                  "incident_update",
                  {
                      "camera_id": event["camera_id"],
                      "event_id": event_id,
                      "evidence_revision": result["evidence_revision"],
                  },
              )
          return {
              "event_id": event_id,
              "status": "promoted" if result else "kept_better_cover",
              "width": width,
              "height": height,
              "candidates": len(assets),
          }
''')
text = text[:start] + new_process
path.write_text(text)

test_path = Path("tests/test_native_evidence.py")
tests = test_path.read_text()
tests = tests.replace(
    "from survng.app.native_evidence import Candidate, NativeEvidenceService, image_quality, shortlist",
    "from survng.app.native_evidence import Candidate, NativeEvidenceService, calibration_epochs, image_quality, shortlist",
    1,
)

old_test = strip_indent('''def test_optional_calibration_failure_does_not_block_disabled_mode_promotion(tmp_path):
          from survng.app.config import CameraConfig
          service, events, event, image, obj, tracking = fixture(tmp_path)
          service.config.detector.native.verification_enabled = False
          service.config.cameras = [CameraConfig(id='test', name='Test', stream_url='rtsp://unused.invalid', native_same_field_of_view=True)]
          tracking.update(state='complete', native_session='test-session')
          events.update_object_tracking(event['id'], tracking)
          service.read_frame = Mock(return_value=cv2.resize(image,(1280,720)))
          service.verifier.detect = Mock(side_effect=RuntimeError('unavailable'))
          assert service.process(event['id'], [Candidate(100,image,[obj],5), Candidate(101,image,[obj],5)])['status'] == 'promoted'
          service.verifier.detect.assert_called_once()
          assert service.counts['calibration_unavailable'] == 1
''')
new_test = strip_indent('''def test_optional_calibration_failure_keeps_substream_when_timing_unverified(tmp_path):
          from survng.app.config import CameraConfig
          service, events, event, image, obj, tracking = fixture(tmp_path)
          service.config.detector.native.verification_enabled = False
          service.config.cameras = [CameraConfig(id='test', name='Test', stream_url='rtsp://unused.invalid', native_same_field_of_view=True)]
          tracking.update(state='complete', native_session='test-session')
          events.update_object_tracking(event['id'], tracking)
          service.read_frame = Mock(return_value=cv2.resize(image,(1280,720)))
          service.match_main = Mock(side_effect=AssertionError('unaligned same-FOV main must not be promoted'))
          service.verifier.detect = Mock(side_effect=RuntimeError('unavailable'))
          result = service.process(event['id'], [Candidate(100,image,[obj],5), Candidate(101,image,[obj],5)])
          assert result['status'] == 'no_usable_candidate'
          service.verifier.detect.assert_called_once()
          service.match_main.assert_not_called()
          assert service.counts['calibration_unavailable'] == 1
          assert events.get(event['id'])['snapshot_path'] == event['snapshot_path']
''')
if old_test not in tests:
    raise SystemExit("optional calibration test anchor not found")
tests = tests.replace(old_test, new_test, 1)

append = strip_indent('''

      def test_calibration_epochs_are_bounded_and_independent_from_cover_shortlist():
          history = [[100 + index * 0.5, 10 + index, 20, 40 + index, 80] for index in range(20)]
          tracking = {'tracks': [{'box_history': history}]}
          epochs = calibration_epochs(tracking)
          assert len(epochs) == 9
          assert epochs[0] == 100
          assert epochs[-1] == 109.5
          assert epochs[-1] - epochs[0] >= 3
          covers = shortlist([Candidate(epoch, None, [], 1000 - epoch) for epoch in epochs])
          assert len(covers) == 3
          assert len(epochs) > len(covers)


      def test_same_fov_calibration_uses_full_track_history_before_cover_promotion(tmp_path, monkeypatch):
          from survng.app.config import CameraConfig

          service, events, event, image, obj, tracking = fixture(tmp_path)
          service.config.detector.native.verification_enabled = False
          service.config.cameras = [CameraConfig(
              id='test', name='Test', stream_url='rtsp://unused.invalid', native_same_field_of_view=True
          )]
          tracking.update(state='complete', native_session='test-session')
          tracking['tracks'][0].update(
              label='person',
              box_history=[
                  [100 + index * 0.5, 100 + index * 4, 80, 240 + index * 4, 300]
                  for index in range(9)
              ],
          )
          events.update_object_tracking(event['id'], tracking)
          main = cv2.resize(image, (1280, 720))
          service.read_frame = Mock(return_value=main)
          service.match_main = Mock(side_effect=AssertionError('aligned same-FOV path must not template-match'))
          service.verifier.detect = Mock(return_value=[
              dict(obj, box={key: value * 2 for key, value in obj['box'].items()})
          ])
          observed = {}

          def fake_alignment(active_tracking, observations):
              observed['count'] = len(observations)
              observed['span'] = observations[-1]['epoch'] - observations[0]['epoch']
              assert active_tracking['native_session'] == 'test-session'
              return {
                  'source': 'main',
                  'verified': True,
                  'offset_seconds': 0.4,
                  'mean_iou': 0.9,
                  'baseline_iou': 0.5,
                  'observations': len(observations),
                  'method': 'test',
              }

          monkeypatch.setattr('survng.app.native_evidence.estimate_replay_alignment', fake_alignment)
          result = service.process(event['id'], [Candidate(100, image, [obj], 5)])
          assert result['status'] == 'promoted'
          assert observed['count'] >= 5
          assert observed['span'] >= 3
          service.match_main.assert_not_called()
          stored = json.loads(events.get(event['id'])['objects_json'])
          cover = next(item for item in stored if item.get('label'))
          assert cover['native_alignment']['method'] == 'same_fov_timestamp_aligned'
          assert cover['native_alignment']['recording_offset_seconds'] == 0.4


      def test_same_fov_unverified_timing_never_promotes_projected_main_without_verification(tmp_path):
          from survng.app.config import CameraConfig

          service, events, event, image, obj, tracking = fixture(tmp_path)
          service.config.detector.native.verification_enabled = False
          service.config.cameras = [CameraConfig(
              id='test', name='Test', stream_url='rtsp://unused.invalid', native_same_field_of_view=True
          )]
          tracking.update(state='complete', native_session='test-session')
          tracking['tracks'][0]['label'] = 'person'
          events.update_object_tracking(event['id'], tracking)
          service.read_frame = Mock(return_value=cv2.resize(image, (1280, 720)))
          service.match_main = Mock(side_effect=AssertionError('unaligned same-FOV main must not be projected'))
          service.verifier.detect = Mock(return_value=[
              dict(obj, box={key: value * 2 for key, value in obj['box'].items()})
          ])

          result = service.process(event['id'], [Candidate(100, image, [obj], 5)])

          assert result['status'] == 'no_usable_candidate'
          service.match_main.assert_not_called()
          assert events.get(event['id'])['snapshot_path'] == event['snapshot_path']
          assert service.counts['replay_alignment_unverified'] == 1
''')
tests += append
test_path.write_text(tests)
