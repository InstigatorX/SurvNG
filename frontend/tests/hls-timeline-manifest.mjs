import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import { mergeRecordingAvailability, recordingPlaybackTimeline } from "../src/recordingPlayback.mjs";

const source = readFileSync(new URL("../src/timeline/TimelinePages.jsx", import.meta.url), "utf8");
const urls = readFileSync(new URL("../src/shared/mediaUrls.js", import.meta.url), "utf8");
const section = (text, start, end) => text.slice(text.indexOf(start), text.indexOf(end)).replace(/^export /, "");
let memo;
let memoDependencies;
const context = vm.createContext({
  URLSearchParams, appUrl: path => path,
  useMemo(create, dependencies) {
    if (!memoDependencies || dependencies.some((value, index) => !Object.is(value, memoDependencies[index]))) {
      memo = create();
      memoDependencies = dependencies;
    }
    return memo;
  },
  desiredEpochRef: { current: 120 },
  dayStart: 0, dayEnd: 86400, date: "2026-09-24", today: "2026-09-24",
  playbackDetail: { start: 100, end: 1000, revision: 1 },
  transport: "hls", useSegmentPlayback: false, isAllCameras: false,
  activeCameraId: "gate", source: "main", manifestRetryToken: 0,
});
context.recordingPlaybackTimeline = recordingPlaybackTimeline;
vm.runInContext(section(source, "  function epochToPlaybackMediaTime(", "  function windowAround("), context);
vm.runInContext(section(urls, "export function recordingDayHlsUrl(", "export function recordingSegmentUrl("), context);
context.playbackTimeline = context.recordingPlaybackTimeline([{ start_epoch: 100, end_epoch: 1000 }]);
let availability = [{ start_epoch: 100, end_epoch: 1000 }];
context.timeline = context.recordingPlaybackTimeline(availability);
const manifestCode = section(source, "  const manifestStartTime =", "  const nativeSegmentUrl =");
const render = () => vm.runInContext(`(() => { ${manifestCode}; return manifestUrl; })()`, context);
const start = url => Number(new URL(url, "http://fixture").searchParams.get("start"));
const initial = render();
assert.equal(start(initial), 20);

// Playback and same-window scrubs update the desired position. New footage at
// the end of the day must not turn either into a new media-resource request.
for (const epoch of [127, 400, 400]) {
  context.desiredEpochRef.current = epoch;
  availability = mergeRecordingAvailability(availability, [{ start_epoch: 1000, end_epoch: 1010 }]);
  context.timeline = context.recordingPlaybackTimeline(availability);
  assert.equal(render(), initial, "availability polling must retain the playlist while playing, scrubbing, or paused");
}

// Midnight changes which date is "today", but does not change the loaded day.
context.today = "2026-09-25";
assert.equal(render(), initial, "midnight must not reload the active archive playlist");

context.manifestRetryToken += 1;
const retry = render();
assert.notEqual(retry, initial, "an explicit retry must still replace the source");
assert.equal(start(retry), 300, "retry must start at the retained position");

context.transport = "original";
context.useSegmentPlayback = true;
assert.equal(render(), "");
context.desiredEpochRef.current = 500;
context.transport = "hls";
context.useSegmentPlayback = false;
assert.equal(start(render()), 400, "returning to HLS must capture the latest playback position");

context.desiredEpochRef.current = 1120;
context.playbackDetail = { start: 1000, end: 1900, revision: 2 };
context.playbackTimeline = context.recordingPlaybackTimeline([{ start_epoch: 1000, end_epoch: 1900 }]);
const nextWindow = render();
assert.equal(start(nextWindow), 120, "a new window must use its requested starting offset");
assert.equal(new URL(nextWindow, "http://fixture").searchParams.get("start_epoch"), "1000.000");

context.activeCameraId = "yard";
context.playbackTimeline = [];
assert.equal(render(), "", "an unloaded camera must not retain the outgoing playlist");
context.desiredEpochRef.current = null;
context.playbackTimeline = context.recordingPlaybackTimeline([{ start_epoch: 1000, end_epoch: 1900 }]);
assert.equal(start(render()), 0, "without a retained position, start at the loaded window's first recording");

// Growing playlist: the player appends recordings and the page extends its
// rows. Neither that nor the moving playhead may change the playlist URL.
context.nativeScope = "yard:main";
context.manifestStartRef = { current: null };
context.desiredEpochRef.current = 1009;
const growingRows = [{ start_epoch: 1000, end_epoch: 1010 }];
context.playbackDetail = { start: 1000, end: 1900, revision: 3, growing: true };
context.playbackTimeline = context.recordingPlaybackTimeline(growingRows);
const growing = render();
assert.equal(new URL(growing, "http://fixture").searchParams.get("growing"), "true");
assert.equal(start(growing), 9);
context.desiredEpochRef.current = 1018;
context.playbackDetail = { ...context.playbackDetail };
context.playbackTimeline = context.recordingPlaybackTimeline([...growingRows, { start_epoch: 1010, end_epoch: 1020 }]);
assert.equal(render(), growing, "extending a growing playlist's rows must not reload it");
context.playbackDetail = { ...context.playbackDetail, revision: 4 };
context.playbackTimeline = context.recordingPlaybackTimeline([...growingRows, { start_epoch: 1010, end_epoch: 1020 }]);
assert.equal(start(render()), 18, "a new load starts at the retained position");

console.log("Timeline playlist stability, retry, transport, and window transition tests passed");
