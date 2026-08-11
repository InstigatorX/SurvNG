import assert from "node:assert/strict";
import { insertZonePoint, insertZonePointWithIndex } from "../src/zoneGeometry.mjs";

const square = [
  { x: 0.1, y: 0.1 },
  { x: 0.9, y: 0.1 },
  { x: 0.9, y: 0.9 },
  { x: 0.1, y: 0.9 },
];

const topPoint = { x: 0.55, y: 0.1 };
assert.deepEqual(
  insertZonePoint(square, topPoint, { x: 640, y: 360 }),
  [square[0], topPoint, square[1], square[2], square[3]],
);
const insertedTopPoint = insertZonePointWithIndex(square, topPoint, { x: 640, y: 360 });
assert.equal(insertedTopPoint.insertionIndex, 1);
assert.deepEqual(
  insertedTopPoint.points.filter((_, index) => index !== insertedTopPoint.insertionIndex),
  square,
);

const leftPoint = { x: 0.1, y: 0.45 };
assert.deepEqual(
  insertZonePoint(square, leftPoint, { x: 640, y: 360 }),
  [...square, leftPoint],
);

const first = { x: 0.2, y: 0.2 };
const second = { x: 0.8, y: 0.2 };
const third = { x: 0.8, y: 0.8 };
assert.deepEqual(insertZonePoint([], first), [first]);
assert.deepEqual(insertZonePoint([first], second), [first, second]);
assert.deepEqual(insertZonePoint([first, second], third), [first, second, third]);

assert.deepEqual(square, [
  { x: 0.1, y: 0.1 },
  { x: 0.9, y: 0.1 },
  { x: 0.9, y: 0.9 },
  { x: 0.1, y: 0.9 },
]);

console.log("zone geometry tests passed");
