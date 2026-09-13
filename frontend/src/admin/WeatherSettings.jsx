import React from "react";
import "../live/weather.css";

export function WeatherSettings({ value, onChange }) {
  const weather = { enabled: false, name: "Local weather", latitude: null, longitude: null, units: "imperial", radar_zoom: 6, animate: true, ...value };
  const update = (key, value) => onChange({ ...weather, [key]: value });
  return <fieldset className="weather-settings">
    <legend>Live weather tile</legend>
    <label><input type="checkbox" checked={weather.enabled} onChange={(event) => update("enabled", event.target.checked)} /> Show weather and radar in Live</label>
    <p className="settings-help">Set the location for all viewers. Conditions come from Open-Meteo; radar and map images load from RainViewer and OpenStreetMap. These services receive requests for this area. Free weather services are intended for personal/noncommercial use.</p>
    {weather.enabled ? <>
      <label>Location name<input maxLength={80} required value={weather.name} onChange={(event) => update("name", event.target.value)} /></label>
      <label>Latitude<input type="number" min={-85} max={85} step="any" required value={weather.latitude ?? ""} onChange={(event) => update("latitude", event.target.value === "" ? null : Number(event.target.value))} /></label>
      <label>Longitude<input type="number" min={-180} max={180} step="any" required value={weather.longitude ?? ""} onChange={(event) => update("longitude", event.target.value === "" ? null : Number(event.target.value))} /></label>
      <label>Units<select value={weather.units} onChange={(event) => update("units", event.target.value)}><option value="imperial">°F · mph</option><option value="metric">°C · km/h</option></select></label>
      <label>Radar zoom<select value={weather.radar_zoom} onChange={(event) => update("radar_zoom", Number(event.target.value))}>{[2, 3, 4, 5, 6, 7].map((zoom) => <option key={zoom} value={zoom}>{zoom}{zoom === 7 ? " — closest" : ""}</option>)}</select></label>
      <label><input type="checkbox" checked={weather.animate} onChange={(event) => update("animate", event.target.checked)} /> Animate recent radar</label>
    </> : null}
  </fieldset>;
}
