import assert from "node:assert/strict";
import { recordingGridLayout } from "../src/recordingGrid.mjs";
import { uniformLiveGridLayout } from "../src/liveWorkspace.mjs";

const cameras = [
  ...Array.from({ length: 8 }, (_, index) => ({
    id: `wide-${index}`,
    stream_dimensions: { live: { width: 1920, height: 1080 } },
  })),
  ...Array.from({ length: 3 }, (_, index) => ({
    id: `four-three-${index}`,
    stream_dimensions: { live: { width: 1280, height: 960 } },
  })),
  {
    id: "portrait",
    stream_dimensions: { live: { width: 672, height: 896 } },
  },
];

const desktop = recordingGridLayout(
  cameras,
  "live",
  1400,
  850,
  8,
  {},
  { portraitPriority: true, portraitRowSpan: 2 },
);
assert.equal(desktop.length, cameras.length);
desktop.forEach((item) => {
  assert.ok(item.x >= 0 && item.y >= 0);
  assert.ok(item.x + item.width <= 1400.001);
  assert.ok(item.y + item.height <= 850.001);
});

const portrait = desktop.find((item) => item.camera.id === "portrait");
assert.ok(Math.abs(portrait.width / portrait.height - 672 / 896) < 0.001);
const landscape = desktop.find((item) => item.camera.id === "wide-0");
assert.ok(portrait.height > landscape.height, "portrait cameras should receive additional height");
assert.ok(
  Math.abs(portrait.height - (landscape.height * 2 + 8)) < 0.01,
  "portrait cameras should span exactly two landscape rows",
);

const unprioritized = recordingGridLayout(cameras, "live", 1400, 850, 8);
const unprioritizedPortrait = unprioritized.find((item) => item.camera.id === "portrait");
assert.ok(
  portrait.width * portrait.height > unprioritizedPortrait.width * unprioritizedPortrait.height,
  "portrait priority should materially increase the portrait camera's displayed area",
);

const resized = recordingGridLayout(
  cameras,
  "live",
  1000,
  700,
  8,
  {},
  { portraitPriority: true, portraitRowSpan: 2 },
);
assert.equal(resized.length, cameras.length);
assert.notDeepEqual(
  resized.map(({ x, y, width, height }) => [x, y, width, height]),
  desktop.map(({ x, y, width, height }) => [x, y, width, height]),
);

const measured = recordingGridLayout(cameras, "live", 1400, 850, 8, { portrait: 1 });
const measuredPortrait = measured.find((item) => item.camera.id === "portrait");
assert.ok(Math.abs(measuredPortrait.width / measuredPortrait.height - 1) < 0.001);

const uniform = uniformLiveGridLayout(cameras, 1400, 850, 4);
assert.equal(uniform.length, cameras.length);
uniform.forEach((item) => {
  assert.equal(item.width, uniform[0].width);
  assert.equal(item.height, uniform[0].height);
  assert.ok(Math.abs(item.width / item.height - (16 / 9)) < 0.001);
  assert.ok(item.x >= 0 && item.y >= 0);
  assert.ok(item.x + item.width <= 1400.001);
  assert.ok(item.y + item.height <= 850.001);
});
assert.equal(new Set(uniform.map((item) => item.width)).size, 1);
assert.equal(new Set(uniform.map((item) => item.height)).size, 1);

const four = uniformLiveGridLayout(cameras.slice(0, 4), 1200, 700, 4);
assert.equal(new Set(four.map((item) => item.y)).size, 2);
assert.equal(new Set(four.map((item) => item.x)).size, 2);
assert.deepEqual(uniformLiveGridLayout([], 1200, 700), []);

console.log("live viewport grid layout tests passed");
