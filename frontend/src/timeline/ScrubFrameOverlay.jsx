import React, { useEffect, useRef } from "react";
import { Film } from "lucide-react";

// Shown over the player while the timeline playhead is dragged: the frame for
// the time under the pointer, cut from a preview strip. The player keeps its
// own picture underneath and takes over again once it has seeked there.
export function ScrubFrameOverlay({ frame }) {
  const canvasRef = useRef(null);
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
  if (!frame) return null;
  // While the first strip of a drag loads there is nothing to draw yet: the
  // player's own picture stays visible. Later the last frame drawn stays up.
  return (
    <div
      className={`recording-scrub-frame${frame.loading ? " loading" : ""}${frame.gap ? " gap" : ""}`}
      data-drawn={strip ? "true" : "false"}
      aria-hidden="true"
    >
      <canvas ref={canvasRef} />
      {frame.gap ? <span><Film size={22} />No recording</span> : null}
    </div>
  );
}
