import assert from "node:assert/strict";
import { camerasWithLiveFraming, DEFAULT_LIVE_FRAMING, liveFramingStyle, normalizedLiveFraming } from "../src/liveFraming.mjs";

assert.deepEqual(normalizedLiveFraming({}, "live"), DEFAULT_LIVE_FRAMING);
assert.deepEqual(normalizedLiveFraming({
  live_view: {
    main: { fit: "contain", focal_x: 12.5, focal_y: 87, zoom: 1.75 },
    live: { fit: "cover", focal_x: 68, focal_y: 31, zoom: 2.25 },
  },
}, "main"), { fit: "contain", focalX: 12.5, focalY: 87, zoom: 1.75 });
assert.deepEqual(normalizedLiveFraming({
  live_view: { live: { fit: "invalid", focal_x: -5, focal_y: 150, zoom: 9 } },
}, "sub"), { fit: "cover", focalX: 0, focalY: 100, zoom: 3 });
assert.deepEqual(liveFramingStyle({
  live_view: { live: { focal_x: 64, focal_y: 42, zoom: 1.4 } },
}), {
  "--live-object-fit": "cover",
  "--live-object-position": "64% 42%",
  "--live-view-zoom": "1.4",
});
const statuses = [{ id: "gate", running: true }, { id: "yard", running: false, live_view: { live: { zoom: 1.2 } } }];
const configured = [{ id: "gate", live_view: { live: { focal_x: 70 } } }];
const merged = camerasWithLiveFraming(statuses, configured);
assert.equal(merged[0].running, true);
assert.equal(merged[0].live_view.live.focal_x, 70);
assert.equal(merged[1].live_view.live.zoom, 1.2);
assert.notEqual(merged[0], statuses[0]);

console.log("live framing tests passed");
