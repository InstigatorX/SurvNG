import React, { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { GripVertical, Maximize2, Pause, Play, Radar, X } from "lucide-react";
import { fetch } from "../shared/api.js";
import { useVisiblePolling } from "../visibilityPolling.mjs";
import { weatherCondition, weatherIsStale, weatherMapTiles } from "../weather.mjs";
import "./weather.css";

const RADAR_HOST = "https://tilecache.rainviewer.com";
const number = (value) => Number.isFinite(value) ? Math.round(value) : "—";
const timeLabel = (timestamp, timeZone) => timestamp ? new Date(timestamp * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", timeZone }) : "—";

function RadarLayer({ tiles, frame, zoom }) {
  const [loaded, setLoaded] = useState(() => new Set());
  const [failed, setFailed] = useState(false);
  const ready = loaded.size === tiles.length;
  return <>
    <div className="weather-map-layer" style={{ opacity: ready && !failed ? 0.72 : 0 }}>
      {tiles.map((tile) => <img key={`${tile.x}:${tile.y}`} alt="" draggable="false" width="256" height="256"
        style={{ left: tile.left, top: tile.top }}
        src={`${RADAR_HOST}${frame.path}/256/${zoom}/${tile.x}/${tile.y}/2/1_1.png`}
        onLoad={() => setLoaded((previous) => new Set([...previous, `${tile.x}:${tile.y}`]))}
        onError={() => setFailed(true)} />)}
    </div>
    {failed || !ready ? <span className="weather-map-message">{failed ? "Radar imagery unavailable" : "Loading radar imagery…"}</span> : null}
  </>;
}

function WeatherMap({ config, frame, zoom }) {
  const ref = useRef(null);
  const [scale, setScale] = useState(1);
  const [mapError, setMapError] = useState(false);
  const [coverageError, setCoverageError] = useState(false);
  const tiles = useMemo(() => weatherMapTiles(config.latitude, config.longitude, zoom), [config.latitude, config.longitude, zoom]);
  useEffect(() => {
    const observer = new ResizeObserver(([entry]) => setScale(Math.max(entry.contentRect.width, entry.contentRect.height) / 512));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => { setMapError(false); setCoverageError(false); }, [tiles]);
  return <div ref={ref} className="weather-map" aria-label={`Regional precipitation radar around ${config.name}`}>
    <div className="weather-map-origin" style={{ transform: `scale(${scale})` }}>
      <div className="weather-map-layer">{tiles.map((tile) => <img key={`${zoom}:${tile.x}:${tile.y}`} alt="" draggable="false" width="256" height="256"
        style={{ left: tile.left, top: tile.top }} referrerPolicy="strict-origin-when-cross-origin" src={`https://tile.openstreetmap.org/${zoom}/${tile.x}/${tile.y}.png`} onError={() => setMapError(true)} />)}</div>
      {frame ? <RadarLayer key={`${frame.path}:${zoom}`} tiles={tiles} frame={frame} zoom={zoom} /> : null}
      <div className="weather-map-layer weather-coverage">{tiles.map((tile) => <img key={`${zoom}:${tile.x}:${tile.y}`} alt="" draggable="false" width="256" height="256"
        style={{ left: tile.left, top: tile.top }} src={`${RADAR_HOST}/v2/coverage/0/256/${zoom}/${tile.x}/${tile.y}/0/0_0.png`} onError={() => setCoverageError(true)} />)}</div>
    </div>
    <span className="weather-location-marker" title={config.name} />
    {mapError || coverageError ? <span className="weather-map-warning">{mapError ? "Basemap unavailable. " : ""}{coverageError ? "Radar coverage unknown." : ""}</span> : null}
  </div>;
}

export function WeatherTile({ camera, timeZone, layout, customLayout, customStyle, onMakePrimary, onAspectChange, dragHandleProps, resizeHandleProps, mobilePrimary, focusPrimary, resizing }) {
  const config = camera.weather;
  const tileRef = useRef(null);
  const dialogRef = useRef(null);
  const wasExpanded = useRef(false);
  const [expanded, setExpanded] = useState(false);
  const [visible, setVisible] = useState(true);
  const [documentVisible, setDocumentVisible] = useState(!document.hidden);
  const [current, setCurrent] = useState(null);
  const [radar, setRadar] = useState(null);
  const [now, setNow] = useState(Date.now() / 1000);
  const [playing, setPlaying] = useState(() => config.animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const [frameIndex, setFrameIndex] = useState(-1);
  const [zoom, setZoom] = useState(config.radar_zoom);
  const active = documentVisible && (visible || expanded);
  const restartKey = `${config.latitude}:${config.longitude}:${config.units}`;
  useEffect(() => { setCurrent(null); setRadar(null); setFrameIndex(-1); }, [restartKey]);
  useEffect(() => { setZoom(config.radar_zoom); }, [config.radar_zoom]);
  useEffect(() => { setPlaying(config.animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches); }, [config.animate]);
  useEffect(() => { onAspectChange?.(camera.id, 16 / 9); }, [camera.id, onAspectChange]);
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting));
    observer.observe(tileRef.current);
    const visibility = () => setDocumentVisible(!document.hidden);
    document.addEventListener("visibilitychange", visibility);
    return () => { observer.disconnect(); document.removeEventListener("visibilitychange", visibility); };
  }, []);
  useVisiblePolling(async (signal) => {
    setNow(Date.now() / 1000);
    await Promise.all([["conditions", setCurrent], ["radar", setRadar]].map(async ([path, setter]) => {
      try {
        const response = await fetch(`/api/weather/${path}`, { signal });
        if (!response.ok) throw new Error("Weather request failed");
        const result = await response.json();
        if (!signal.aborted) setter(result);
      } catch {
        if (!signal.aborted) setter((previous) => ({ ...previous, stale: true, unavailable: !previous?.data }));
      }
    }));
  }, 60_000, active, { restartKey });
  const frames = radar?.data?.frames || [];
  const index = frameIndex < 0 ? Math.max(0, frames.length - 1) : Math.min(frameIndex, frames.length - 1);
  const frame = frames[index];
  useEffect(() => {
    if (!active || !playing || frames.length < 2) return undefined;
    const timer = window.setInterval(() => setFrameIndex((previous) => (previous + 1) % frames.length), 1200);
    return () => window.clearInterval(timer);
  }, [active, playing, frames.length]);
  useEffect(() => {
    if (expanded) dialogRef.current?.showModal();
    else if (wasExpanded.current) tileRef.current?.querySelector('[aria-label="Expand weather"]')?.focus();
    wasExpanded.current = expanded;
  }, [expanded]);
  const data = current?.data;
  const [icon, condition] = weatherCondition(data?.weather_code, data?.is_day);
  const unit = config.units === "imperial" ? "°F" : "°C";
  const windUnit = config.units === "imperial" ? "mph" : "km/h";
  const conditionsStale = weatherIsStale(current, now, 1200) || Boolean(data && now - data.time > 3600);
  const radarStale = weatherIsStale(radar, now, 900) || Boolean(frames.length && now - frames[frames.length - 1].time > 1800);
  const content = <div className="weather-content">
    <WeatherMap config={config} frame={active ? frame : null} zoom={zoom} />
    <header className="weather-heading">
      <strong><Radar size={16} /> {config.name}</strong>
      <div className="weather-actions">
        {customLayout && !expanded ? <button type="button" className="camera-drag-handle" {...dragHandleProps}><GripVertical size={16} /></button> : null}
        {onMakePrimary && !expanded ? <button type="button" onClick={() => onMakePrimary(camera)} title="Make primary weather view" aria-label="Make primary weather view">◎</button> : null}
        <button type="button" onClick={() => setExpanded(!expanded)} aria-label={expanded ? "Close weather" : "Expand weather"}>{expanded ? <X size={16} /> : <Maximize2 size={16} />}</button>
      </div>
    </header>
    <div className="weather-conditions">
      {data ? <><span className="weather-temperature"><span aria-hidden="true">{icon}</span> {number(data.temperature_2m)}{unit}</span><strong>{condition}</strong>
        <span className="weather-details">Feels {number(data.apparent_temperature)}° · Humidity {number(data.relative_humidity_2m)}%</span>
        <span className="weather-details">Wind {number(data.wind_speed_10m)} · Gusts {number(data.wind_gusts_10m)} {windUnit}</span>
        <small>{conditionsStale ? "Stale · " : ""}Conditions {timeLabel(data.time, timeZone)}</small></> : <span role="status">{current?.unavailable ? "Conditions unavailable · retrying" : "Loading conditions…"}</span>}
    </div>
    <footer className="weather-footer">
      <div className="weather-radar-controls">
        <button type="button" onClick={() => setPlaying(!playing)} disabled={frames.length < 2} aria-label={playing ? "Pause radar" : "Play radar"}>{playing ? <Pause size={15} /> : <Play size={15} />}</button>
        <input aria-label="Radar frame" type="range" min="0" max={Math.max(0, frames.length - 1)} value={Math.max(0, index)} disabled={!frames.length} onChange={(event) => { setPlaying(false); setFrameIndex(Number(event.target.value)); }} />
        <span>{frame ? `${radarStale ? "Stale · " : ""}Radar ${timeLabel(frame.time, timeZone)}` : radar?.unavailable ? "Radar unavailable · retrying" : "Loading radar…"}</span>
        {expanded ? <select aria-label="Radar zoom" value={zoom} onChange={(event) => setZoom(Number(event.target.value))}>{[2, 3, 4, 5, 6, 7].map((level) => <option key={level} value={level}>Zoom {level}</option>)}</select> : null}
      </div>
      <div className="weather-attribution"><span>Shaded: no radar coverage</span><a href="https://open-meteo.com/" target="_blank" rel="noreferrer">Open-Meteo</a><a href="https://www.rainviewer.com/" target="_blank" rel="noreferrer">Weather data by RainViewer</a><a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a></div>
    </footer>
  </div>;
  return <>
    <article ref={tileRef} data-camera-id={camera.id} className={`bento-card camera-tile weather-tile ${layout ? "viewport-layout" : ""} ${customLayout ? "custom-layout-tile" : ""} ${mobilePrimary ? "mobile-primary" : ""} ${focusPrimary ? "focus-primary" : ""} ${resizing ? "resizing" : ""}`}
      style={customLayout ? customStyle : layout ? { left: layout.x, top: layout.y, width: layout.width, height: layout.height } : undefined}>
      {expanded ? <button className="weather-expanded-placeholder" onClick={() => setExpanded(false)}>Close expanded weather</button> : content}
      {customLayout ? <button type="button" className="camera-resize-handle" {...resizeHandleProps} /> : null}
    </article>
    {expanded ? createPortal(<dialog ref={dialogRef} className="weather-dialog" aria-label={`${config.name} weather and radar`} onCancel={() => setExpanded(false)} onClose={() => setExpanded(false)}>
      {content}
      <div className="weather-hourly"><strong>Next hours</strong>{data?.hourly?.length ? data.hourly.map((hour) => <div key={hour.time}><span>{timeLabel(hour.time, timeZone)}</span><strong>{number(hour.temperature_2m)}{unit}</strong><span>{number(hour.precipitation_probability)}% precip.</span></div>) : <span>Hourly forecast unavailable</span>}</div>
    </dialog>, document.body) : null}
  </>;
}
