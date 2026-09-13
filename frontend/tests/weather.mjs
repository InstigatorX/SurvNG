import assert from "node:assert/strict";
import { liveItemsWithWeather, weatherCondition, weatherIsStale, weatherMapTiles } from "../src/weather.mjs";
import { readLiveCustomLayout } from "../src/liveCustomLayout.mjs";
import { liveDensityPage, orderedLiveCamerasForFocus } from "../src/liveWorkspace.mjs";

const cameras = [{ id: "door" }, { id: "weather:local" }];
assert.equal(liveItemsWithWeather(cameras, { enabled: false }), cameras);
const items = liveItemsWithWeather(cameras, { enabled: true, name: "Home" });
assert.equal(items.length, 3);
assert.equal(new Set(items.map((item) => item.id)).size, 3);
const weather = items[2];
const layout = readLiveCustomLayout("{}", items, {});
assert.ok(layout.sizes[weather.id]);
assert.ok(layout.order.includes(weather.id));
assert.equal(orderedLiveCamerasForFocus(items, weather.id, true)[0], weather);
assert.equal(liveDensityPage(items, "fit", 0).cameras[2], weather);
assert.equal(readLiveCustomLayout(JSON.stringify(layout), cameras, {}).order.includes(weather.id), false);
assert.deepEqual(weatherCondition(0, 0), ["☾", "Clear"]);
assert.equal(weatherCondition(75)[1], "Snow");
assert.equal(weatherCondition(null)[1], "Conditions unavailable");
for (const longitude of [-180, -74, 180]) {
  const tiles = weatherMapTiles(40, longitude, 6);
  assert.equal(tiles.length, 9);
  assert.ok(tiles.every((tile) => tile.x >= 0 && tile.x < 64 && tile.y >= 0 && tile.y < 64));
  assert.equal(tiles.filter((tile) => tile.left <= 0 && tile.left + 256 > 0 && tile.top <= 0 && tile.top + 256 > 0).length, 1);
}
assert.ok(weatherMapTiles(85, 0, 2).every((tile) => tile.y >= 0));
assert.equal(weatherIsStale({ updated_at: 1000, stale: false }, 1100, 600), false);
assert.equal(weatherIsStale({ updated_at: 1000, stale: false }, 1700, 600), true);
assert.equal(weatherIsStale({ updated_at: 1000, stale: true }, 1001, 600), true);
console.log("weather and mixed live layout tests passed");
