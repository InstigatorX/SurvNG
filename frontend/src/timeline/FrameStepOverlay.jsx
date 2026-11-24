import React, { useSyncExternalStore } from "react";
import { StepBack, StepForward } from "lucide-react";
import { formatFrameTime } from "../frameStepping.mjs";

function useFrameStep(stepper) {
  return useSyncExternalStore(stepper.subscribe, stepper.getSnapshot, stepper.getSnapshot);
}

const NOTICES = {
  end: "No recording after this frame",
  start: "No recording before this frame",
  error: "Could not load this frame",
};

// The stepped frame over the paused player: the exact frame the server
// decoded, which stays up until stepping ends (play, a seek, a scrub, another
// camera). Until the first frame has loaded the player's own picture shows.
export function FrameStepOverlay({ stepper }) {
  const state = useFrameStep(stepper);
  const notice = state.active ? NOTICES[state.notice] || "" : "";
  return (
    <>
      {/* Present before anything is shown, so that its text is announced. */}
      <span className="visually-hidden" role="status">{notice}</span>
      {state.active ? (
        <div
          className={`recording-frame-step${state.loading ? " loading" : ""}`}
          data-drawn={state.image ? "true" : "false"}
          data-epoch={Number.isFinite(state.shownEpoch) ? state.shownEpoch.toFixed(6) : ""}
          aria-hidden="true"
        >
          {state.image ? <img src={state.image} alt="" draggable="false" /> : null}
          {notice ? <span className="recording-frame-step-notice">{notice}</span> : null}
        </div>
      ) : null}
    </>
  );
}

// The time readout: the stepped frame's time to the millisecond while
// stepping, else what it would show anyway (`children`).
export function FrameStepTime({ stepper, timeZone, prefix = "", children }) {
  const state = useFrameStep(stepper);
  if (!state.active || !Number.isFinite(state.epoch)) return children;
  return <span className="timeline-frame-time">{prefix}{formatFrameTime(state.epoch, timeZone)}</span>;
}

// One frame back / forward. A press steps once; holding repeats. Keyboard
// activation (Enter, Space) steps once per press.
export function FrameStepButtons({ stepper, disabled }) {
  return [[-1, "Previous frame", ",", StepBack], [1, "Next frame", ".", StepForward]].map(([direction, label, key, Icon]) => (
    <button
      key={direction}
      type="button"
      className="timeline-hud-frame"
      disabled={disabled}
      aria-label={label}
      title={`${label} (${key})`}
      onPointerDown={(event) => {
        if (disabled || (event.pointerType === "mouse" && event.button !== 0)) return;
        // Keep the press while a finger drifts off the button.
        try { event.currentTarget.setPointerCapture?.(event.pointerId); } catch { /* not capturable */ }
        stepper.press(direction);
      }}
      onPointerUp={() => stepper.release()}
      onPointerCancel={() => stepper.release()}
      onLostPointerCapture={() => stepper.release()}
      onBlur={() => stepper.release()}
      onClick={(event) => {
        // Pointer presses stepped on pointerdown; this is the keyboard.
        if (event.detail === 0) stepper.step(direction);
      }}
      onContextMenu={(event) => event.preventDefault()}
    >
      <Icon size={16} aria-hidden="true" />
    </button>
  ));
}
