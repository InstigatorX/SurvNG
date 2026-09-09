import React, { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { supportsNativeRecordingHls } from "../recordingPlayback.mjs";
import { fetch } from "./api.js";
import { ShakaVideo } from "./media.jsx";

// Let Safari own the playlist and buffer across recording boundaries. Other
// browsers use Shaka's MSE path; neither path requests a video transcode.
export const RecordingHlsVideo = forwardRef(function RecordingHlsVideo({
  src, startTime, bufferingGoal, mimeType = "application/vnd.apple.mpegurl",
  onReady, onError, ...videoProps
}, forwardedRef) {
  const videoRef = useRef(null);
  const callbacks = useRef({ onReady, onError });
  callbacks.current = { onReady, onError };
  const [nativeHls] = useState(supportsNativeRecordingHls);
  useImperativeHandle(forwardedRef, () => videoRef.current);

  useEffect(() => {
    if (!nativeHls) return undefined;
    const video = videoRef.current;
    const controller = new AbortController();
    let checking = false;
    let disposed = false;
    let timer;
    const ready = () => callbacks.current.onReady?.(null, video);
    const failed = async () => {
      if (checking) return;
      checking = true;
      const error = { code: video.error?.code, message: video.error?.message || "HLS playback failed" };
      // Safari also reports format errors for inaccessible playlists. Do not
      // start an encoder to work around authentication or a failed request.
      if (error.code === 3 || error.code === 4) {
        timer = window.setTimeout(() => controller.abort(), 10_000);
        try {
          const response = await fetch(src, { signal: controller.signal });
          if (!response.ok || !(await response.text()).trimStart().startsWith("#EXTM3U")) {
            error.category = 1;
            error.message = `Recording playlist unavailable (${response.status})`;
          }
        } catch {
          error.category = 1;
          error.message = "Recording playlist request failed";
        } finally {
          window.clearTimeout(timer);
        }
      }
      if (!disposed && videoRef.current === video && video.getAttribute("src") === src) {
        callbacks.current.onError?.(error);
      }
      checking = false;
    };
    video.addEventListener("loadedmetadata", ready);
    video.addEventListener("error", failed);
    // React's development effect replay may have released this same node.
    if (video.getAttribute("src") !== src) video.setAttribute("src", src);
    return () => {
      disposed = true;
      video.removeEventListener("loadedmetadata", ready);
      video.removeEventListener("error", failed);
      controller.abort();
      window.clearTimeout(timer);
      // Release the captured outgoing resource, never the replacement ref.
      video.pause();
      video.removeAttribute("src");
      video.load();
    };
  }, [nativeHls, src]);

  return nativeHls
    // Safari can retain the previous HLS window's buffered ranges and remain
    // seeking forever when its source changes. Reset only at playlist/window
    // changes; all recording fragments within that playlist share one player.
    ? <video key={src} {...videoProps} ref={videoRef} src={src} />
    : <ShakaVideo {...videoProps} ref={videoRef} src={src} mimeType={mimeType}
      startTime={startTime} bufferingGoal={bufferingGoal} onReady={onReady} onError={onError} />;
});
