export function liveItemsWithWeather(cameras, weather) {
  if (!weather?.enabled) return cameras;
  let id = "weather:local";
  while (cameras.some((camera) => camera.id === id)) id += ":weather";
  return [...cameras, { id, name: weather.name || "Local weather", kind: "weather", weather }];
}

export function weatherCondition(code, isDay = 1) {
  if (code == null) return ["◌", "Conditions unavailable"];
  if (code === 0) return [isDay ? "☀" : "☾", "Clear"];
  if (code <= 3) return ["☁", code === 3 ? "Overcast" : "Partly cloudy"];
  if ([45, 48].includes(code)) return ["≋", "Fog"];
  if (code >= 51 && code <= 57) return ["☂", "Drizzle"];
  if ((code >= 71 && code <= 77) || [85, 86].includes(code)) return ["❄", "Snow"];
  if (code >= 95) return ["ϟ", "Thunderstorm"];
  if ((code >= 61 && code <= 67) || (code >= 80 && code <= 82)) return ["☂", "Rain"];
  return ["◌", "Unknown conditions"];
}

// A fixed regional map, using the same Web Mercator coordinates for every layer.
// Nine tiles cover a centered 512px viewport even across the antimeridian.
export function weatherMapTiles(latitude, longitude, zoom) {
  const count = 2 ** zoom;
  const lat = Math.max(-85, Math.min(85, latitude)) * Math.PI / 180;
  const x = (longitude + 180) / 360 * count;
  const y = (1 - Math.asinh(Math.tan(lat)) / Math.PI) / 2 * count;
  const tiles = [];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const tx = Math.floor(x) + dx;
      const ty = Math.floor(y) + dy;
      if (ty < 0 || ty >= count) continue;
      tiles.push({ x: ((tx % count) + count) % count, y: ty, left: (tx - x) * 256, top: (ty - y) * 256 });
    }
  }
  return tiles;
}

export function weatherIsStale(result, now, maxAge) {
  return Boolean(result && (result.stale || !result.updated_at || now - result.updated_at > maxAge));
}
