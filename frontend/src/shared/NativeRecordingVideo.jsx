import React, { forwardRef, useLayoutEffect, useRef, useState } from "react";

// Keep display ownership separate from playback controls while a new source
// loads. The outgoing decoded frame survives until the requested frame is ready.
export const NativeRecordingVideo = forwardRef(function NativeRecordingVideo(
  { src, nextSrc = "", muted = true, playbackRate = 1, ...callbacks }, forwardedRef,
) {
  const videos = useRef([null, null]);
  const sources = useRef(["", ""]);
  const active = useRef(0);
  const displayed = useRef(null);
  const requested = useRef("");
  const metadataDelivered = useRef(false);
  const endedDelivered = useRef(false);
  const props = useRef(null);
  props.current = { muted, playbackRate, ...callbacks };
  const [visible, setVisible] = useState(null);

  function expose(video) {
    if (typeof forwardedRef === "function") forwardedRef(video);
    else if (forwardedRef) forwardedRef.current = video;
  }

  function ownsEvents(index) {
    return Boolean(requested.current) && active.current === index && sources.current[index] === requested.current;
  }

  function forward(name, index) {
    if (ownsEvents(index)) props.current[name]?.({ currentTarget: videos.current[index] });
  }

  function metadata(index) {
    if (!ownsEvents(index) || metadataDelivered.current || videos.current[index].readyState < 1) return;
    metadataDelivered.current = true;
    forward("onLoadedMetadata", index);
  }

  function reveal(index) {
    const video = videos.current[index];
    if (!ownsEvents(index) || !metadataDelivered.current || video.readyState < 2 || video.seeking) return;
    displayed.current = index;
    setVisible(index);
  }

  function load(index, url) {
    const video = videos.current[index];
    video.pause();
    video.muted = true;
    sources.current[index] = url;
    if (url) video.src = url;
    else video.removeAttribute("src");
    video.load();
  }

  useLayoutEffect(() => {
    if (requested.current === src) return;
    requested.current = src;
    metadataDelivered.current = false;
    endedDelivered.current = false;
    const warm = sources.current.indexOf(src);
    const index = warm >= 0 ? warm : displayed.current === 0 ? 1 : 0;
    // Suppress outgoing pause/time/ended events before changing either element.
    active.current = index;
    videos.current[1 - index].pause();
    videos.current[1 - index].muted = true;
    if (sources.current[index] !== src || videos.current[index].error) load(index, src);
    const video = videos.current[index];
    video.muted = props.current.muted;
    video.playbackRate = props.current.playbackRate;
    expose(video);
    // A preloaded element already emitted this event while it was standby.
    if (video.readyState >= 1) metadata(index);
    reveal(index);
  }, [src]);

  useLayoutEffect(() => {
    const video = videos.current[active.current];
    video.muted = muted;
    video.playbackRate = playbackRate;
  }, [muted, playbackRate]);

  useLayoutEffect(() => {
    // Never recycle the outgoing frame while the incoming source is loading.
    if (displayed.current !== active.current || requested.current !== src) return;
    const spare = 1 - active.current;
    const url = nextSrc && nextSrc !== src ? nextSrc : "";
    if (sources.current[spare] !== url) load(spare, url);
  }, [src, nextSrc, visible]);

  useLayoutEffect(() => () => {
    requested.current = "";
    for (const video of videos.current) {
      video?.pause();
      video?.removeAttribute("src");
      video?.load();
    }
    sources.current = ["", ""];
    expose(null);
  }, []);

  return <>{[0, 1].map((index) => <video
    key={index}
    ref={(video) => { videos.current[index] = video; }}
    className={visible === index ? "active" : "standby"}
    aria-hidden={visible !== index}
    playsInline
    preload="auto"
    onLoadedMetadata={() => metadata(index)}
    onLoadedData={() => reveal(index)}
    onCanPlay={() => reveal(index)}
    onPlaying={() => reveal(index)}
    onSeeked={() => { forward("onSeeked", index); reveal(index); }}
    onTimeUpdate={() => forward("onTimeUpdate", index)}
    onPlay={() => {
      if (ownsEvents(index) && !videos.current[index].ended) endedDelivered.current = false;
      forward("onPlay", index);
    }}
    onPause={() => forward("onPause", index)}
    onError={() => { if (videos.current[index].error) forward("onError", index); }}
    onEnded={() => {
      if (!ownsEvents(index) || endedDelivered.current || !videos.current[index].ended) return;
      endedDelivered.current = true;
      forward("onEnded", index);
    }}
  />)}</>;
});
