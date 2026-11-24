import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { Film } from "lucide-react";

// A frame that would draw the same picture is not worth a render.
function sameFrame(a, b) {
  return a === b || Boolean(
    a && b && a.strip === b.strip && a.gap === b.gap && a.loading === b.loading
    && a.tile?.index === b.tile?.index,
  );
}

// Shown over the player while a timeline is scrubbed: the frame for the time
// under the playhead, cut from a preview strip. The player keeps its own
// picture underneath and takes over again once it has seeked there.
//
// Frames arrive through the ref (`show(frame, options)`), not through props:
// a scrub reports a position on every scroll event, and only this overlay
// should render for it, not the page that owns the player.
export const ScrubFrameOverlay = forwardRef(function ScrubFrameOverlay({ seeking = false }, ref) {
  const canvasRef = useRef(null);
  const [frame, setFrame] = useState(null);
  const [released, setReleased] = useState(false);
  const clearTimerRef = useRef(null);
  const drawnRef = useRef(false);

  useImperativeHandle(ref, () => ({
    show(next, { committed = false } = {}) {
      if (next?.held) {
        // The released time's own frame, arriving after the release. It
        // replaces the picture only while that is still up.
        setFrame((current) => (current && !sameFrame(current, next) ? next : current));
        return;
      }
      if (clearTimerRef.current) window.clearTimeout(clearTimerRef.current);
      clearTimerRef.current = null;
      if (next) {
        setReleased(false);
        setFrame((current) => (sameFrame(current, next) ? current : next));
        return;
      }
      if (!committed) {
        setReleased(false);
        setFrame(null);
        return;
      }
      // After a release the last frame stays up until the player has seeked
      // there, so the picture does not jump back to where it was. Not
      // released at once: the seek that follows starts a moment later (a
      // rail commits after its scroll settles), and until it has started
      // "not seeking" would be taken for "arrived".
      clearTimerRef.current = window.setTimeout(() => {
        setReleased(true);
        clearTimerRef.current = window.setTimeout(() => {
          clearTimerRef.current = null;
          setReleased(false);
          setFrame(null);
        }, 4000);
      }, 300);
    },
  }), []);

  useEffect(() => {
    if (!released || seeking) return;
    if (clearTimerRef.current) window.clearTimeout(clearTimerRef.current);
    clearTimerRef.current = null;
    setReleased(false);
    setFrame(null);
  }, [released, seeking]);

  useEffect(() => () => {
    if (clearTimerRef.current) window.clearTimeout(clearTimerRef.current);
  }, []);

  const strip = frame?.strip || null;
  const tile = frame?.tile || null;
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !strip?.image || !tile) return;
    const width = Math.max(1, Math.round(tile.sw));
    const height = Math.max(1, Math.round(tile.sh));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(strip.image, tile.sx, tile.sy, tile.sw, tile.sh, 0, 0, width, height);
  }, [strip, tile?.sx, tile?.sy, tile?.sw, tile?.sh]);

  // Until something has been drawn the player's own picture stays visible.
  // After that the last frame drawn stays up while the next one loads.
  if (!frame) drawnRef.current = false;
  else if (strip && tile) drawnRef.current = true;
  if (!frame) return null;
  return (
    <div
      className={`recording-scrub-frame${frame.loading ? " loading" : ""}${frame.gap ? " gap" : ""}`}
      data-drawn={drawnRef.current ? "true" : "false"}
      aria-hidden="true"
    >
      <canvas ref={canvasRef} />
      {frame.gap ? <span><Film size={22} />No recording</span> : null}
    </div>
  );
});
