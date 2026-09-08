import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { NativeRecordingVideo } from "../../src/shared/NativeRecordingVideo.jsx";
import "../../src/styles.css";

const events = [];
const ids = new WeakMap();
let nextId = 0;
const id = (video) => video && (ids.get(video) || (ids.set(video, ++nextId), nextId));

function Fixture() {
  const video = useRef(null);
  const [config, configure] = useState({ src: "", nextSrc: "", muted: true, playbackRate: 1, seek: 0, playing: false, advance: false });
  const current = useRef(config);
  const [status, setStatus] = useState("");
  current.current = config;
  function record(name, event) {
    const target = event.currentTarget;
    events.push({ name, id: id(target), src: target.getAttribute("src"), time: target.currentTime, ownsRef: target === video.current });
  }
  window.harness = {
    configure: (patch) => configure((value) => ({ ...value, ...patch })),
    events,
    video: () => video.current,
    id,
    snapshot: () => [...document.querySelectorAll("video")].map((element) => ({
      id: id(element), src: element.getAttribute("src"), activeRef: element === video.current,
      visible: getComputedStyle(element).visibility === "visible", ready: element.readyState,
      paused: element.paused, muted: element.muted, time: element.currentTime, seeking: element.seeking,
    })),
  };
  useEffect(() => {
    const timer = setInterval(() => setStatus(JSON.stringify(window.harness.snapshot(), null, 2)), 200);
    return () => clearInterval(timer);
  }, []);
  return <><div id="stage" className="recordings-v2-player" style={{ position: "relative", width: 320, height: 180, minHeight: 180 }}>
    <NativeRecordingVideo ref={video} src={config.src} nextSrc={config.nextSrc} muted={config.muted} playbackRate={config.playbackRate}
      onLoadedMetadata={(event) => {
        record("metadata", event);
        event.currentTarget.currentTime = current.current.seek;
        if (current.current.playing) event.currentTarget.play().catch(() => {});
      }}
      onSeeked={(event) => record("seeked", event)}
      onTimeUpdate={(event) => record("timeupdate", event)}
      onPlay={(event) => record("play", event)}
      onPause={(event) => record("pause", event)}
      onError={(event) => record("error", event)}
      onEnded={(event) => {
        record("ended", event);
        if (current.current.advance && current.current.nextSrc) configure((value) => ({ ...value, src: value.nextSrc, nextSrc: "", seek: 0 }));
      }}
    />
  </div>
    <button onClick={() => configure({ src: "/clips/red.mp4", nextSrc: "/clips/green.mp4", muted: true, playbackRate: 1, seek: 0, playing: false, advance: false })}>Load first and preload next</button>
    <button onClick={() => {
      configure((value) => ({ ...value, playing: true, advance: true }));
      video.current.currentTime = video.current.duration - .2;
      video.current.play();
    }}>Play through boundary</button>
    <button onClick={async () => {
      await fetch("/test/hold", { method: "POST" });
      configure((value) => ({ ...value, src: "/clips/blue-delayed.mp4", nextSrc: "/clips/red-next.mp4", seek: .75, playing: false, advance: false }));
    }}>Hold next clip</button>
    <button onClick={() => fetch("/test/release", { method: "POST" })}>Release next clip</button>
    <button onClick={() => configure((value) => ({ ...value, src: "/clips/green-latest.mp4", nextSrc: "", seek: .6, playing: false, advance: false }))}>Supersede with green</button>
    <button onClick={() => configure((value) => ({ ...value, src: "/clips/broken.mp4", nextSrc: "", playing: false, advance: false }))}>Fail next clip</button>
    <pre aria-label="Video buffer status">{status}</pre>
  </>;
}

createRoot(document.getElementById("root")).render(<Fixture />);
