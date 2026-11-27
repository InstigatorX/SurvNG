import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { CarFront, ChevronDown, ChevronUp, ChevronsDown, Footprints, Radar, ScanSearch, UserRound } from "lucide-react";
import { eventThumbnailUrl, recordingPreviewUrl } from "../shared/mediaUrls.js";
import { formatDuration } from "../shared/format.js";
import { useViewportQuery } from "../shared/hooks.js";
import {
  RAIL_SPANS, condenseStory, crossCameraLinks, formatQuietDuration, layoutRailCards, railDensityBuckets,
  railEventDurationSeconds, railEventKind, railEventTitle, railTickSeconds,
} from "../timelineRail.mjs";

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

function clockFormatter(timeZone, seconds) {
  return new Intl.DateTimeFormat("en-US", { timeZone, hour: "numeric", minute: "2-digit", ...(seconds ? { second: "2-digit" } : {}) });
}

/** "2:30", "2 PM" on the hour; the period is shown only on hours to save space. */
function useTickLabel(timeZone) {
  return useMemo(() => {
    const parts = new Intl.DateTimeFormat("en-US", { timeZone, hour: "numeric", minute: "2-digit", hourCycle: "h12" });
    return (epoch) => {
      const fields = Object.fromEntries(parts.formatToParts(new Date(epoch * 1000)).map((part) => [part.type, part.value]));
      return fields.minute === "00" ? `${fields.hour} ${fields.dayPeriod || ""}`.trim() : `${fields.hour}:${fields.minute}`;
    };
  }, [timeZone]);
}

function useElementHeight(ref) {
  const [height, setHeight] = useState(0);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return undefined;
    setHeight(element.clientHeight);
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(() => setHeight(element.clientHeight));
    observer.observe(element);
    return () => observer.disconnect();
  }, [ref]);
  return height;
}

const SCRUB_HOLD_MS = 3000;
const TOUCH_STALE_MS = 4000;

/**
 * Scroll-to-scrub along a vertical time axis, newest at the top. The playhead
 * line stays fixed at `lineRatio` of the viewport; native scrolling (momentum
 * on touch, wheel on desktop) moves time under it. A scroll that settles for
 * 180 ms commits a seek.
 */
export function useVerticalScrub({ topEpoch, bottomEpoch, spanSeconds, playhead, onSeek, onScrub = null, lineRatio = 0.42 }) {
  const scrollerRef = useRef(null);
  const height = useElementHeight(scrollerRef);
  const pps = height > 0 ? height / Math.max(60, spanSeconds) : 0;
  const lineY = Math.round(height * lineRatio);
  const contentHeight = Math.max(height, Math.round((topEpoch - bottomEpoch) * pps + height));
  const [scrollTop, setScrollTop] = useState(0);
  const [scrubEpoch, setScrubEpoch] = useState(null);
  // Only scrolling while the user is interacting (touch, wheel, drag, keys or
  // the edge rail) scrubs; layout changes and playback-driven scrolling never seek.
  const intentUntilRef = useRef(0);
  // A touch counts while its finger is down. touchend can be lost when the card
  // under the finger unmounts mid-gesture, so a touch with no activity for
  // TOUCH_STALE_MS is treated as ended rather than freezing the rail.
  const touchingRef = useRef(false);
  const touchActivityRef = useRef(0);
  const touchActive = useCallback(() => touchingRef.current && performance.now() - touchActivityRef.current < TOUCH_STALE_MS, []);
  const markIntent = useCallback((ms = 1200) => { intentUntilRef.current = Math.max(intentUntilRef.current, performance.now() + ms); }, []);
  const commitTimerRef = useRef(null);
  const holdTimerRef = useRef(null);
  const pendingRef = useRef(null);
  // scrollTop this hook set itself; the scroll event it causes is not a scrub.
  const programmaticTopRef = useRef(null);
  const [holdReleased, setHoldReleased] = useState(0);
  // The committed scrub time shown while playback catches up to it.
  const [heldEpoch, setHeldEpoch] = useState(null);
  const onSeekRef = useRef(onSeek);
  onSeekRef.current = onSeek;
  // Told the time under the line as it is scrubbed, then null when the
  // scrub ends, so the player can show that frame meanwhile.
  const onScrubRef = useRef(onScrub);
  onScrubRef.current = onScrub;
  const scrubReportedRef = useRef(false);
  const y = useCallback((epoch) => lineY + (topEpoch - epoch) * pps, [lineY, pps, topEpoch]);
  const epochAtScroll = useCallback((top) => clamp(topEpoch - top / Math.max(pps, 1e-9), bottomEpoch, topEpoch), [bottomEpoch, pps, topEpoch]);

  useLayoutEffect(() => {
    const element = scrollerRef.current;
    if (!element || !pps || scrubEpoch !== null || touchActive() || !Number.isFinite(playhead)) return;
    const pending = pendingRef.current;
    // Hold the committed scrub position until playback reports the new time;
    // a seek that snapped elsewhere is released by the hold timer.
    if (pending && Math.abs(playhead - pending.epoch) > 2 && Date.now() - pending.at < SCRUB_HOLD_MS) return;
    pendingRef.current = null;
    setHeldEpoch(null);
    const target = Math.max(0, (topEpoch - clamp(playhead, bottomEpoch, topEpoch)) * pps);
    if (Math.abs(element.scrollTop - target) < 0.5) return;
    element.scrollTop = target;
    programmaticTopRef.current = element.scrollTop;
    setScrollTop(element.scrollTop);
  }, [bottomEpoch, holdReleased, playhead, pps, scrubEpoch, topEpoch, touchActive]);

  useEffect(() => () => {
    window.clearTimeout(commitTimerRef.current);
    window.clearTimeout(holdTimerRef.current);
    if (scrubReportedRef.current) onScrubRef.current?.(null, { committed: false });
    scrubReportedRef.current = false;
  }, []);

  const onScroll = useCallback((event) => {
    const top = event.currentTarget.scrollTop;
    setScrollTop(top);
    const programmatic = programmaticTopRef.current !== null && Math.abs(top - programmaticTopRef.current) < 1;
    programmaticTopRef.current = null;
    if (touchingRef.current && !programmatic) touchActivityRef.current = performance.now();
    if (programmatic || (!touchActive() && performance.now() > intentUntilRef.current)) return;
    markIntent(400); // momentum keeps scrubbing until it settles
    const epoch = epochAtScroll(top);
    setScrubEpoch(epoch);
    scrubReportedRef.current = true;
    onScrubRef.current?.(epoch);
    window.clearTimeout(commitTimerRef.current);
    commitTimerRef.current = window.setTimeout(() => {
      pendingRef.current = { epoch, at: Date.now() };
      setHeldEpoch(epoch);
      setScrubEpoch(null);
      scrubReportedRef.current = false;
      onScrubRef.current?.(null, { committed: true });
      onSeekRef.current?.(epoch);
      window.clearTimeout(holdTimerRef.current);
      holdTimerRef.current = window.setTimeout(() => setHoldReleased((count) => count + 1), SCRUB_HOLD_MS + 50);
    }, 180);
  }, [epochAtScroll, markIntent, touchActive]);

  // An explicit jump (edge rail, keys) is a user scrub, so it is not marked programmatic.
  const scrollToEpoch = useCallback((epoch) => {
    const element = scrollerRef.current;
    if (!element || !pps) return;
    markIntent();
    element.scrollTop = Math.max(0, (topEpoch - clamp(epoch, bottomEpoch, topEpoch)) * pps);
  }, [bottomEpoch, markIntent, pps, topEpoch]);

  // Mouse users can grab the rail as well as wheel it.
  const dragRef = useRef(null);
  const onPointerDown = useCallback((event) => {
    if (event.pointerType !== "mouse" || event.button !== 0 || event.target.closest("button,a,input")) return;
    markIntent();
    dragRef.current = { y: event.clientY, top: event.currentTarget.scrollTop };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }, [markIntent]);
  const onPointerMove = useCallback((event) => {
    if (!dragRef.current) return;
    markIntent();
    event.currentTarget.scrollTop = dragRef.current.top - (event.clientY - dragRef.current.y);
  }, [markIntent]);
  // pointercancel fires when native touch panning takes over; touch intent is
  // tracked by the touch events instead, so only the mouse drag ends here.
  const onPointerUp = useCallback(() => { dragRef.current = null; }, []);
  const onWheel = useCallback(() => markIntent(), [markIntent]);
  const onTouchStart = useCallback(() => { touchingRef.current = true; touchActivityRef.current = performance.now(); }, []);
  const onTouchMove = useCallback(() => { touchActivityRef.current = performance.now(); }, []);
  const onTouchEnd = useCallback(() => {
    touchingRef.current = false;
    markIntent(600); // a released touch may still fling; momentum scroll events extend this
  }, [markIntent]);

  const visible = useMemo(() => {
    if (!pps) return { start: bottomEpoch, end: topEpoch };
    const toEpoch = (contentY) => topEpoch - (contentY - lineY) / pps;
    return { start: Math.max(bottomEpoch, toEpoch(scrollTop + height * 2)), end: Math.min(topEpoch, toEpoch(scrollTop - height)) };
  }, [bottomEpoch, height, lineY, pps, scrollTop, topEpoch]);

  // The line reads the video's own time; scrollTop is whole (or half) pixels,
  // which at an hour per screen is several seconds, so it is only the fallback.
  const lineEpoch = scrubEpoch ?? heldEpoch ?? (Number.isFinite(playhead) ? clamp(playhead, bottomEpoch, topEpoch) : (pps ? epochAtScroll(scrollTop) : null));
  return {
    scrollerRef, height, pps, lineY, contentHeight, y, visible, scrubEpoch, lineEpoch, scrollToEpoch, scrollTop,
    scrollerProps: { ref: scrollerRef, onScroll, onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp, onWheel, onTouchStart, onTouchMove, onTouchEnd, onTouchCancel: onTouchEnd },
  };
}

const EVENT_FILTERS = [["all", "All"], ["people", "People"], ["vehicles", "Vehicles"], ["motion", "Motion"]];
const QUIET_MINUTES = [2, 5, 15, 30];

function KindIcon({ kind, size = 18 }) {
  if (kind === "person") return <UserRound size={size} aria-hidden="true" />;
  if (kind === "vehicle") return <CarFront size={size} aria-hidden="true" />;
  if (kind === "motion") return <Footprints size={size} aria-hidden="true" />;
  return <ScanSearch size={size} aria-hidden="true" />;
}

function scrubKeyHandler(lineEpoch, spanSeconds, bottomEpoch, topEpoch, scrollToEpoch) {
  return (event) => {
    const step = { ArrowUp: spanSeconds / 30, ArrowDown: -spanSeconds / 30, PageUp: spanSeconds / 2, PageDown: -spanSeconds / 2 }[event.key];
    if (step) { event.preventDefault(); scrollToEpoch(lineEpoch + step); return; }
    if (event.key === "Home") { event.preventDefault(); scrollToEpoch(topEpoch); }
    if (event.key === "End") { event.preventDefault(); scrollToEpoch(bottomEpoch); }
  };
}

/** Option A: the vertical Event Rail. */
export function EventRail({
  cameraId, source, dayStart, dayEnd, topEpoch, playhead, recordings, events, densityEvents,
  spanMinutes, onSpanChange, filter, onFilterChange, selectedEventId, onEventSelect, onSeek, timeZone,
  onPreviousDay = null, onNextDay = null, cameraNames = null, onScrub = null,
}) {
  const spanSeconds = spanMinutes * 60;
  const scrub = useVerticalScrub({ topEpoch, bottomEpoch: dayStart, spanSeconds, playhead, onSeek, onScrub });
  const { y, visible, pps, lineY, contentHeight, lineEpoch } = scrub;
  const tickLabel = useTickLabel(timeZone);
  const bubble = useMemo(() => clockFormatter(timeZone, true), [timeZone]);
  const short = useMemo(() => clockFormatter(timeZone, false), [timeZone]);
  const ticks = useMemo(() => {
    if (!pps) return [];
    const step = railTickSeconds(pps);
    const items = [];
    for (let epoch = Math.ceil(visible.start / step) * step; epoch <= visible.end; epoch += step) items.push(epoch);
    return items;
  }, [pps, visible.end, visible.start]);
  const coverage = useMemo(() => (recordings || [])
    .filter((range) => range.end_epoch >= visible.start && range.start_epoch <= visible.end)
    .map((range) => {
      const end = Math.min(range.end_epoch, topEpoch);
      return { key: range.start_epoch, top: y(end), height: Math.max(2, (end - range.start_epoch) * pps) };
    }), [pps, recordings, topEpoch, visible.end, visible.start, y]);
  const orderedRecordings = useMemo(() => [...(recordings || [])].sort((left, right) => left.start_epoch - right.start_epoch), [recordings]);
  const gaps = useMemo(() => {
    const ordered = orderedRecordings;
    const items = [];
    for (let index = 1; index < ordered.length; index++) {
      const start = ordered[index - 1].end_epoch;
      const end = ordered[index].start_epoch;
      if (end > visible.start && start < visible.end && (end - start) * pps >= 18) items.push({ key: start, top: y(end) + ((end - start) * pps) / 2 - 8 });
    }
    return items;
  }, [orderedRecordings, pps, visible.end, visible.start, y]);
  const marks = useMemo(() => events
    .filter((event) => event.incident_epoch >= visible.start && event.incident_epoch <= visible.end)
    .map((event) => {
      const duration = railEventDurationSeconds(event);
      const height = Math.max(6, duration * pps);
      return { key: event.id, kind: railEventKind(event), top: y(event.incident_epoch) - height, height };
    }), [events, pps, visible.end, visible.start, y]);
  const cardEvents = useMemo(() => (filter === "motion" ? events : events.filter((event) => event.has_objects)), [events, filter]);
  // Lay cards out once per zoom; scrolling only changes which are mounted.
  const allCards = useMemo(() => (pps ? layoutRailCards(cardEvents, y) : []), [cardEvents, pps, y]);
  const cards = useMemo(() => allCards
    .filter((card) => card.top + 60 >= y(visible.end) - 60 && card.top <= y(visible.start) + 60), [allCards, visible.end, visible.start, y]);
  const keyHandler = scrubKeyHandler(lineEpoch, spanSeconds, dayStart, topEpoch, scrub.scrollToEpoch);
  const filters = EVENT_FILTERS;

  return (
    <aside className="event-rail" aria-label="Event timeline">
      <div className="event-rail-controls">
        <div className="event-rail-heading">
          <h2>Timeline</h2>
          <div className="event-rail-segmented" role="group" aria-label="Rail zoom">
            {RAIL_SPANS.map((span) => (
              <button key={span.minutes} type="button" aria-pressed={spanMinutes === span.minutes} onClick={() => onSpanChange(span.minutes)}>{span.label}</button>
            ))}
          </div>
        </div>
        <div className="event-rail-filters" role="group" aria-label="Event type">
          {filters.map(([value, label]) => (
            <button key={value} type="button" aria-pressed={filter === value} onClick={() => onFilterChange(value)}>{label}</button>
          ))}
        </div>
      </div>
      <div className="event-rail-body">
        <div
          className="event-rail-scroller"
          {...scrub.scrollerProps}
          tabIndex={0}
          role="group"
          aria-label="Scrub recording time"
          aria-description="Scroll, or use the arrow, Page and Home/End keys, to move the playhead; events inside are buttons."
          onKeyDown={keyHandler}
        >
          <div className="event-rail-content" style={{ height: contentHeight }}>
            {ticks.map((epoch) => (
              <div key={epoch} className="event-rail-tick" style={{ top: y(epoch) }}>
                {/* The playhead bubble owns the label slot at the line. */}
                {Math.abs(y(epoch) - scrub.scrollTop - lineY) > 24 ? <span>{tickLabel(epoch)}</span> : null}
              </div>
            ))}
            {coverage.map((segment) => <div key={segment.key} className="event-rail-coverage" style={{ top: segment.top, height: segment.height }} />)}
            {gaps.map((gap) => <div key={gap.key} className="event-rail-gap" style={{ top: gap.top }}>No recording</div>)}
            {marks.map((mark) => <div key={mark.key} className={`event-rail-mark ${mark.kind}`} style={{ top: mark.top, height: mark.height }} />)}
            {cards.map((card) => {
              const selected = card.events.some((event) => String(event.id) === String(selectedEventId));
              const kind = railEventKind(card.event);
              const duration = railEventDurationSeconds(card.event);
              const thumbnail = card.event.snapshot_url || card.event.snapshot_path ? eventThumbnailUrl(card.event, 160, 72, { objectFocus: true, aspectWidth: 16, aspectHeight: 10 }) : "";
              return (
                <button
                  key={card.event.id}
                  type="button"
                  className={`event-rail-card ${kind}${selected ? " selected" : ""}`}
                  style={{ top: card.top }}
                  aria-pressed={selected}
                  onClick={() => onEventSelect(card.event)}
                >
                  <span className="event-rail-thumb">
                    {thumbnail ? <img src={thumbnail} alt="" loading="lazy" decoding="async" onError={(event) => { event.currentTarget.hidden = true; }} /> : null}
                    <KindIcon kind={kind} />
                  </span>
                  <span className="event-rail-card-text">
                    <strong>{railEventTitle(card.event)}</strong>
                    <small>{cameraNames?.[card.event.camera_id] ? `${cameraNames[card.event.camera_id]} · ` : ""}{short.format(new Date(card.event.incident_epoch * 1000))}{duration ? ` · ${formatDuration(duration)}` : ""}</small>
                  </span>
                  {card.events.length > 1 ? <span className="event-rail-more">+{card.events.length - 1}</span> : null}
                </button>
              );
            })}
          </div>
        </div>
        <div className="event-rail-playhead" style={{ top: lineY }}>
          <span>{Number.isFinite(lineEpoch) ? bubble.format(new Date(lineEpoch * 1000)).replace(/\s?[AP]M$/, "") : "--"}</span>
        </div>
        <EdgeRail
          cameraId={cameraId}
          source={source}
          onScrub={onScrub}
          dayStart={dayStart}
          dayEnd={dayEnd}
          topEpoch={topEpoch}
          events={densityEvents || events}
          recordings={recordings}
          windowStart={lineEpoch - (lineY / Math.max(pps, 1e-9))}
          windowEnd={lineEpoch + ((scrub.height - lineY) / Math.max(pps, 1e-9))}
          playhead={lineEpoch}
          onSeek={scrub.scrollToEpoch}
          timeZone={timeZone}
          onPreviousDay={onPreviousDay}
          onNextDay={onNextDay}
        />
        <p className="event-rail-hint">Scroll or drag to scrub · newest at top</p>
      </div>
    </aside>
  );
}

/** Option B: the 24 h activity strip with a scrubbing loupe. */
export function EdgeRail({ cameraId, source, dayStart, dayEnd, topEpoch, events, recordings, windowStart, windowEnd, playhead, onSeek, timeZone, onPreviousDay = null, onNextDay = null, onScrub = null }) {
  const railRef = useRef(null);
  const [loupe, setLoupe] = useState(null);
  const [previewEpoch, setPreviewEpoch] = useState(null);
  const gestureRef = useRef(null);
  const span = Math.max(1, dayEnd - dayStart);
  const buckets = useMemo(() => railDensityBuckets(dayStart, dayEnd, events, recordings), [dayEnd, dayStart, events, recordings]);
  const label = useMemo(() => clockFormatter(timeZone, false), [timeZone]);
  const fraction = (epoch) => clamp((dayEnd - epoch) / span, 0, 1);
  const epochAt = (clientY) => {
    const rect = railRef.current.getBoundingClientRect();
    return clamp(dayEnd - ((clientY - rect.top) / rect.height) * span, dayStart, Math.min(dayEnd, topEpoch));
  };
  // With a scrub callback the player shows the frame under the loupe, so
  // the loupe itself carries only the time.
  const playerPreview = Boolean(onScrub && cameraId);
  const onScrubRef = useRef(onScrub);
  onScrubRef.current = onScrub;
  // A layout switch can unmount this mid-gesture; the scrub must still end.
  useEffect(() => () => {
    if (gestureRef.current) onScrubRef.current?.(null, { committed: false });
    gestureRef.current = null;
  }, []);
  useEffect(() => {
    if (!loupe || playerPreview) return undefined;
    const timer = window.setTimeout(() => setPreviewEpoch(Math.round(loupe.epoch / 5) * 5), 140);
    return () => window.clearTimeout(timer);
  }, [loupe, playerPreview]);
  const update = (event) => {
    const gesture = gestureRef.current;
    if (!gesture) return;
    const rect = railRef.current.getBoundingClientRect();
    // Sliding away from the rail switches to fine steps of two seconds per pixel.
    const fine = event.clientX < rect.left - 40;
    if (fine && !gesture.fine) Object.assign(gesture, { fine: true, anchorY: event.clientY, anchorEpoch: gesture.epoch });
    if (!fine && gesture.fine) gesture.fine = false;
    const epoch = gesture.fine
      ? clamp(gesture.anchorEpoch - (event.clientY - gesture.anchorY) * 2, dayStart, Math.min(dayEnd, topEpoch))
      : epochAt(event.clientY);
    gesture.epoch = epoch;
    onScrub?.(epoch);
    // The loupe is placed in the edge rail, whose track starts below the next-day button.
    setLoupe({ epoch, y: railRef.current.offsetTop + clamp(event.clientY - rect.top, 0, rect.height), fine: gesture.fine });
  };
  const onPointerDown = (event) => {
    event.preventDefault();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    gestureRef.current = { epoch: epochAt(event.clientY), fine: false };
    update(event);
  };
  const onPointerUp = () => {
    const gesture = gestureRef.current;
    gestureRef.current = null;
    setLoupe(null);
    setPreviewEpoch(null);
    if (gesture) onScrub?.(null, { committed: true });
    if (gesture) onSeek(gesture.epoch);
  };
  const onKeyDown = (event) => {
    const step = { ArrowUp: 900, ArrowDown: -900, PageUp: 3600, PageDown: -3600 }[event.key];
    if (!step) return;
    event.preventDefault();
    onSeek(clamp(playhead + step, dayStart, Math.min(dayEnd, topEpoch)));
  };
  const hours = [0, 6, 12, 18].map((hour) => dayStart + hour * 3600);
  return (
    <div className="edge-rail">
      {/* Newest at the top, so the next day sits above the track and the previous one below. */}
      <button type="button" className="edge-rail-day next" onClick={onNextDay || undefined} disabled={!onNextDay} aria-label="Next day"><ChevronUp size={16} aria-hidden="true" /></button>
      <div
        ref={railRef}
        className="edge-rail-track"
        role="slider"
        tabIndex={0}
        aria-label="Jump within the day"
        aria-orientation="vertical"
        aria-valuemin={Math.round(dayStart)}
        aria-valuemax={Math.round(dayEnd)}
        aria-valuenow={Math.round(playhead || dayStart)}
        aria-valuetext={Number.isFinite(playhead) ? label.format(new Date(playhead * 1000)) : ""}
        onPointerDown={onPointerDown}
        onPointerMove={update}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onKeyDown={onKeyDown}
      >
        {buckets.map((bucket) => bucket.kind === "empty" ? null : (
          <span
            key={bucket.start}
            className={`edge-rail-bucket ${bucket.kind}`}
            style={{ top: `${fraction(bucket.end) * 100}%`, height: `${(900 / span) * 100}%`, width: `${Math.max(14, Math.round(bucket.level * 100))}%` }}
          />
        ))}
        {hours.map((epoch) => <span key={epoch} className="edge-rail-hour" style={{ top: `${fraction(epoch) * 100}%` }}>{label.format(new Date(epoch * 1000)).replace(":00", "").replace(" ", "").toLowerCase().replace("m", "")}</span>)}
        <span className="edge-rail-window" style={{ top: `${fraction(windowEnd) * 100}%`, height: `${Math.max(1.5, ((windowEnd - windowStart) / span) * 100)}%` }} />
      </div>
      <button type="button" className="edge-rail-day previous" onClick={onPreviousDay || undefined} disabled={!onPreviousDay} aria-label="Previous day"><ChevronDown size={16} aria-hidden="true" /></button>
      {loupe ? (
        <div className="edge-rail-loupe" style={{ top: loupe.y }} role="status">
          {playerPreview ? null : cameraId && Number.isFinite(previewEpoch) ? <img src={recordingPreviewUrl(cameraId, previewEpoch, source, { width: 320 })} alt="" /> : <span className="edge-rail-loupe-blank" />}
          <strong>{new Intl.DateTimeFormat("en-US", { timeZone, hour: "numeric", minute: "2-digit", second: loupe.fine ? "2-digit" : undefined }).format(new Date(loupe.epoch * 1000))}</strong>
          <small>{loupe.fine ? "Fine scrub · 2 s per pixel" : "Slide left for fine steps"}</small>
        </div>
      ) : null}
    </div>
  );
}

/** Option C: every camera on one clock. Rows on desktop, columns on a phone. */
export function CameraLanes({
  cameras, recordings, events, source, startEpoch, endEpoch, topEpoch, dayStart, playhead, onSeek, onSelectCamera, timeZone,
  spanMinutes, onSpanChange, windowHours, onWindowHoursChange, onScrub = null,
}) {
  const compact = useViewportQuery("(max-width: 760px)");
  const [activeOnly, setActiveOnly] = useState(false);
  const byCamera = useMemo(() => {
    const map = new Map(cameras.map((camera) => [camera.id, { events: [], recordings: [] }]));
    for (const event of events) map.get(event.camera_id)?.events.push(event);
    // The grid returns coverage for every source; show the one being played.
    for (const range of recordings) if (!range.source || range.source === source) map.get(range.camera_id)?.recordings.push(range);
    return map;
  }, [cameras, events, recordings, source]);
  const lanes = cameras.filter((camera) => !activeOnly || byCamera.get(camera.id)?.events.length);
  const links = useMemo(() => crossCameraLinks(events), [events]);
  if (compact) {
    return <CameraColumns lanes={lanes} byCamera={byCamera} links={links} dayStart={dayStart} topEpoch={topEpoch} playhead={playhead} onSeek={onSeek} onScrub={onScrub} onSelectCamera={onSelectCamera} timeZone={timeZone} spanMinutes={spanMinutes} onSpanChange={onSpanChange} />;
  }
  return <CameraRows lanes={lanes} byCamera={byCamera} links={links} startEpoch={startEpoch} endEpoch={endEpoch} playhead={playhead} onSeek={onSeek} onSelectCamera={onSelectCamera} timeZone={timeZone} activeOnly={activeOnly} onActiveOnlyChange={setActiveOnly} windowHours={windowHours} onWindowHoursChange={onWindowHoursChange} />;
}

function CameraRows({ lanes, byCamera, links, startEpoch, endEpoch, playhead, onSeek, onSelectCamera, timeZone, activeOnly, onActiveOnlyChange, windowHours, onWindowHoursChange }) {
  const span = Math.max(1, endEpoch - startEpoch);
  const tickLabel = useTickLabel(timeZone);
  const bubble = useMemo(() => clockFormatter(timeZone, true), [timeZone]);
  const fraction = (epoch) => clamp((epoch - startEpoch) / span, 0, 1);
  const percent = (epoch) => `${fraction(epoch) * 100}%`;
  const names = new Map(lanes.map((camera) => [camera.id, camera.name || camera.id]));
  const step = span > 12 * 3600 ? 7200 : span > 4 * 3600 ? 3600 : span > 3600 ? 1800 : 600;
  const ticks = [];
  for (let epoch = Math.ceil(startEpoch / step) * step; epoch < endEpoch; epoch += step) ticks.push(epoch);
  const laneIndex = new Map(lanes.map((camera, index) => [camera.id, index]));
  const seekFromTrack = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    onSeek(startEpoch + clamp((event.clientX - rect.left) / rect.width, 0, 1) * span);
  };
  return (
    <section className="camera-lanes" aria-label="Camera lanes">
      <div className="camera-lanes-toolbar">
        <strong>Cameras</strong>
        <label><input type="checkbox" checked={activeOnly} onChange={(event) => onActiveOnlyChange(event.target.checked)} />Only lanes with activity</label>
        <div className="event-rail-segmented" role="group" aria-label="Lane zoom">
          {[[1, "1h"], [4, "4h"], [12, "12h"], [24, "24h"]].map(([hours, label]) => (
            <button key={hours} type="button" aria-pressed={windowHours === hours} onClick={() => onWindowHoursChange(hours)}>{label}</button>
          ))}
        </div>
      </div>
      <div className="camera-lanes-grid">
        <div className="camera-lanes-axis" aria-hidden="true">
          {ticks.map((epoch) => <span key={epoch} style={{ left: percent(epoch) }}>{tickLabel(epoch)}</span>)}
        </div>
        {lanes.map((camera) => {
          const lane = byCamera.get(camera.id) || { events: [], recordings: [] };
          const busy = lane.events.some((event) => Math.abs(event.incident_epoch - playhead) < 90);
          return (
            <div key={camera.id} className={`camera-lane${busy ? " active" : ""}`}>
              <button type="button" className="camera-lane-name" onClick={() => onSelectCamera(camera.id)} title={`Open ${camera.name || camera.id} at this time`}>
                <i />{camera.name || camera.id}
              </button>
              <div className="camera-lane-track" onClick={seekFromTrack} role="presentation">
                {ticks.map((epoch) => <span key={epoch} className="camera-lane-gridline" style={{ left: percent(epoch) }} />)}
                {lane.recordings.filter((range) => range.end_epoch > startEpoch && range.start_epoch < endEpoch).map((range) => (
                  <span key={range.start_epoch} className="camera-lane-coverage" style={{ left: percent(range.start_epoch), width: `calc(${percent(Math.min(range.end_epoch, endEpoch))} - ${percent(range.start_epoch)})` }} />
                ))}
                {lane.events.filter((event) => event.incident_epoch >= startEpoch && event.incident_epoch <= endEpoch).map((event) => (
                  <span
                    key={event.id}
                    className={`camera-lane-mark ${railEventKind(event)}`}
                    title={`${railEventTitle(event)} · ${bubble.format(new Date(event.incident_epoch * 1000))}`}
                    style={{ left: percent(event.incident_epoch), width: `max(6px, ${(railEventDurationSeconds(event) / span) * 100}%)` }}
                  />
                ))}
              </div>
            </div>
          );
        })}
        {links.filter((link) => laneIndex.has(link.from.camera_id) && laneIndex.has(link.to.camera_id) && link.from.incident_epoch >= startEpoch && link.from.incident_epoch <= endEpoch).map((link) => {
          const first = Math.min(laneIndex.get(link.from.camera_id), laneIndex.get(link.to.camera_id));
          const last = Math.max(laneIndex.get(link.from.camera_id), laneIndex.get(link.to.camera_id));
          return (
            <div key={`${link.from.id}-${link.to.id}`} className="camera-lanes-link" style={{ "--at": fraction(link.from.incident_epoch), "--from": first, "--to": last }}>
              <span>Same {railEventKind(link.from) === "vehicle" ? "vehicle" : "person"} · {names.get(link.from.camera_id)} → {names.get(link.to.camera_id)}</span>
            </div>
          );
        })}
        <div className="camera-lanes-playhead" style={{ "--at": fraction(playhead) }} aria-hidden="true"><span>{bubble.format(new Date(playhead * 1000)).replace(/\s?[AP]M$/, "")}</span></div>
      </div>
    </section>
  );
}

function CameraColumns({ lanes, byCamera, dayStart, topEpoch, playhead, onSeek, onScrub = null, onSelectCamera, timeZone, spanMinutes, onSpanChange }) {
  const scrub = useVerticalScrub({ topEpoch, bottomEpoch: dayStart, spanSeconds: spanMinutes * 60, playhead, onSeek, onScrub, lineRatio: 0.45 });
  const { y, visible, pps, lineY, contentHeight, lineEpoch } = scrub;
  const tickLabel = useTickLabel(timeZone);
  const bubble = useMemo(() => clockFormatter(timeZone, false), [timeZone]);
  const ticks = [];
  if (pps) {
    const step = railTickSeconds(pps, 52);
    for (let epoch = Math.ceil(visible.start / step) * step; epoch <= visible.end; epoch += step) ticks.push(epoch);
  }
  return (
    <section className="camera-columns" aria-label="Camera lanes" style={{ "--columns": Math.max(1, lanes.length) }}>
      <div className="camera-columns-heads">
        {lanes.map((camera) => (
          <button key={camera.id} type="button" onClick={() => onSelectCamera(camera.id)} title={`Open ${camera.name || camera.id} at this time`}>{camera.name || camera.id}</button>
        ))}
      </div>
      <div className="event-rail-body">
        <div className="event-rail-scroller" {...scrub.scrollerProps} tabIndex={0} role="group" aria-label="Scrub all cameras"
          onKeyDown={scrubKeyHandler(lineEpoch, spanMinutes * 60, dayStart, topEpoch, scrub.scrollToEpoch)}>
          <div className="event-rail-content camera-columns-content" style={{ height: contentHeight, "--columns": lanes.length }}>
            {ticks.map((epoch) => <div key={epoch} className="event-rail-tick" style={{ top: y(epoch) }}><span>{tickLabel(epoch)}</span></div>)}
            {lanes.map((camera, index) => {
              const lane = byCamera.get(camera.id) || { events: [], recordings: [] };
              return (
                <div key={camera.id} className="camera-column" style={{ "--index": index }}>
                  {lane.recordings.filter((range) => range.end_epoch >= visible.start && range.start_epoch <= visible.end).map((range) => {
                    const end = Math.min(range.end_epoch, topEpoch);
                    return <span key={range.start_epoch} className="camera-column-spine" style={{ top: y(end), height: Math.max(2, (end - range.start_epoch) * pps) }} />;
                  })}
                  {lane.events.filter((event) => event.incident_epoch >= visible.start && event.incident_epoch <= visible.end).map((event) => {
                    const height = Math.max(10, railEventDurationSeconds(event) * pps);
                    return (
                      <span key={event.id} className={`camera-column-mark ${railEventKind(event)}`}
                        style={{ top: y(event.incident_epoch) - height, height }}
                        title={`${camera.name || camera.id}: ${railEventTitle(event)} at ${bubble.format(new Date(event.incident_epoch * 1000))}`} />
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
        <div className="event-rail-playhead" style={{ top: lineY }}><span>{Number.isFinite(lineEpoch) ? bubble.format(new Date(lineEpoch * 1000)).replace(/\s?[AP]M$/, "") : "--"}</span></div>
      </div>
      <div className="event-rail-controls camera-columns-controls">
        <div className="event-rail-segmented" role="group" aria-label="Lane zoom">
          {RAIL_SPANS.map((span) => <button key={span.minutes} type="button" aria-pressed={spanMinutes === span.minutes} onClick={() => onSpanChange(span.minutes)}>{span.label}</button>)}
        </div>
        <small>Tap a camera to open it at this moment</small>
      </div>
    </section>
  );
}

/** Option D: the condensed story. Quiet stretches collapse; events come first. */
export function CatchUpStory({
  events, dayStart, endEpoch, playhead, selectedEventId, onPlayEvent, onShowRealTime, cameraNames, timeZone,
  quietMinutes, onQuietMinutesChange, filter = "all", onFilterChange = null,
}) {
  const items = useMemo(() => condenseStory(events, { quietSeconds: quietMinutes * 60, startEpoch: dayStart, endEpoch }), [dayStart, endEpoch, events, quietMinutes]);
  const short = useMemo(() => clockFormatter(timeZone, false), [timeZone]);
  // Newest at top, like the Timeline rail.
  const ordered = useMemo(() => [...items].reverse(), [items]);
  const selectedRef = useRef(null);
  const listRef = useRef(null);
  useEffect(() => {
    const node = selectedRef.current;
    const list = listRef.current;
    if (!node || !list) return;
    list.scrollTop = Math.max(0, node.offsetTop - list.clientHeight / 2 + node.clientHeight / 2);
  }, [selectedEventId]);
  const eventCount = items.filter((item) => item.type === "event").length;
  const quietTotal = items.filter((item) => item.type === "quiet").reduce((sum, item) => sum + item.seconds, 0);
  return (
    <aside className="catchup-story" aria-label="Catch up">
      <div className="event-rail-controls">
        <div className="event-rail-heading">
          <h2>Catch up</h2>
          <div className="event-rail-segmented" role="group" aria-label="Collapse quiet stretches longer than">
            {QUIET_MINUTES.map((minutes) => (
              <button key={minutes} type="button" aria-pressed={quietMinutes === minutes} title={`Collapse quiet stretches longer than ${minutes} minutes`} onClick={() => onQuietMinutesChange(minutes)}>{minutes}m</button>
            ))}
          </div>
        </div>
        <div className="event-rail-filters" role="group" aria-label="Event type">
          {onFilterChange ? EVENT_FILTERS.map(([value, label]) => (
            <button key={value} type="button" aria-pressed={filter === value} onClick={() => onFilterChange(value)}>{label}</button>
          )) : null}
        </div>
      </div>
      {!eventCount ? <p className="catchup-empty"><Radar size={16} aria-hidden="true" />No events match this filter today.</p> : null}
      <ol ref={listRef} className="catchup-list">
        {ordered.map((item) => {
          if (item.type === "quiet") {
            return (
              <li key={`quiet-${item.start}`} className="catchup-quiet">
                <button type="button" onClick={() => onShowRealTime(item.start)} title="Show this stretch in real time">
                  <ChevronsDown size={14} aria-hidden="true" />{formatQuietDuration(item.seconds)} quiet
                </button>
              </li>
            );
          }
          const event = item.event;
          const kind = railEventKind(event);
          const selected = String(event.id) === String(selectedEventId);
          const playing = playhead >= item.start - 3 && playhead <= item.end + 3;
          const thumbnail = event.snapshot_url || event.snapshot_path ? eventThumbnailUrl(event, 240, 72, { objectFocus: true, aspectWidth: 16, aspectHeight: 10 }) : "";
          const duration = railEventDurationSeconds(event);
          return (
            <li key={event.id} ref={selected ? selectedRef : null} className={`catchup-event ${kind}${selected ? " selected" : ""}`}>
              <button type="button" aria-pressed={selected} onClick={() => onPlayEvent(event)}>
                <span className="catchup-thumb">
                  {thumbnail ? <img src={thumbnail} alt="" loading="lazy" decoding="async" onError={(error) => { error.currentTarget.hidden = true; }} /> : null}
                  <KindIcon kind={kind} size={20} />
                  {playing && selected ? <span className="catchup-progress" style={{ width: `${clamp(((playhead - item.start) / Math.max(1, item.end - item.start)) * 100, 0, 100)}%` }} /> : null}
                </span>
                <span className="catchup-text">
                  <strong>{railEventTitle(event)}</strong>
                  <time>{short.format(new Date(item.start * 1000))}</time>
                  <small>{[cameraNames?.[event.camera_id], duration ? formatDuration(duration) : null, selected ? (playing ? "playing" : "selected") : null].filter(Boolean).join(" · ")}</small>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
      <p className="catchup-summary">{eventCount} events · {formatQuietDuration(quietTotal)} quiet · newest at top</p>
    </aside>
  );
}
