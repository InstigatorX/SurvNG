import { useCallback, useEffect, useState } from "react";
import { Images, Link2, Unlink2, CircleHelp, Trash2, RefreshCw } from "lucide-react";
import { appUrl, fetch } from "../shared/api.js";

function SampleStrip({ samples, title }) {
  if (!samples?.length) return null;
  return (
    <div className="reid-sample-strip">
      <strong>{title}</strong>
      <div className="reid-sample-strip-row">
        {samples.map((sample) => (
          <figure key={sample.sample_id}>
            <img src={appUrl(sample.crop_url)} alt={sample.sample_id} loading="lazy" />
            <figcaption>{sample.camera_id} · {sample.selection_reason || "crop"}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}

export default function ReidTrainingPanel() {
  const [queue, setQueue] = useState({ hard_pairs: [], tracks: [], status: {} });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const loadQueue = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/reid-training/review/queue?limit=12&hours=168");
      if (!response.ok) throw new Error(`Review queue failed (${response.status})`);
      setQueue(await response.json());
    } catch (err) {
      setError(err?.message || "Failed to load ReID review queue");
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void loadQueue();
  }, [loadQueue]);

  const submitReview = async (body) => {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const response = await fetch("/api/reid-training/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || `Review failed (${response.status})`);
      }
      const result = await response.json();
      setMessage(`Recorded ${result.action.replaceAll("_", " ")}`);
      await loadQueue();
    } catch (err) {
      setError(err?.message || "Review action failed");
      setBusy(false);
    }
  };

  const pairs = queue.hard_pairs || [];
  const tracks = queue.tracks || [];
  const status = queue.status || {};

  return (
    <section id="admin-panel-reid" className="bento-card config-editor settings-panel reid-training-panel">
      <div className="settings-panel-header">
        <div>
          <h2>ReID Training</h2>
          <p>
            Review cross-camera appearance pairs to label same/different people for environment fine-tuning.
            Names are optional; anonymous identities are enough.
          </p>
        </div>
        <button type="button" onClick={() => void loadQueue()} disabled={busy}>
          <RefreshCw size={16} /> Refresh
        </button>
      </div>

      <div className="reid-training-status">
        <span>{status.samples ?? 0} samples</span>
        <span>{status.identities ?? 0} identities</span>
        <span>{status.auto_samples ?? 0} auto-labeled</span>
        <span>{status.pair_reviews ?? 0} pair decisions</span>
      </div>

      {error ? <p className="settings-error">{error}</p> : null}
      {message ? <p className="settings-help">{message}</p> : null}

      <h3>Hard pairs</h3>
      {!pairs.length ? (
        <p className="settings-help">
          No unresolved cross-camera pairs with training crops yet. Enable crop collection and let person
          tracks accumulate across cameras.
        </p>
      ) : (
        <div className="reid-pair-list">
          {pairs.map((pair) => {
            const key = `${pair.left.event_id}-${pair.left.track_id}-${pair.right.event_id}-${pair.right.track_id}`;
            const body = {
              left_event_id: pair.left.event_id,
              left_track_id: pair.left.track_id,
              right_event_id: pair.right.event_id,
              right_track_id: pair.right.track_id,
              similarity: pair.similarity,
            };
            return (
              <article className="reid-pair-card" key={key}>
                <header>
                  <strong>
                    {pair.left.camera_id} ↔ {pair.right.camera_id}
                  </strong>
                  <span>
                    similarity {pair.similarity.toFixed(3)}
                    {pair.visually_similar ? " · above threshold" : " · near threshold"}
                  </span>
                </header>
                <div className="reid-pair-columns">
                  <SampleStrip samples={pair.left.samples} title={`Event ${pair.left.event_id} / track ${pair.left.track_id}`} />
                  <SampleStrip samples={pair.right.samples} title={`Event ${pair.right.event_id} / track ${pair.right.track_id}`} />
                </div>
                <div className="reid-pair-actions">
                  <button type="button" disabled={busy} onClick={() => void submitReview({ ...body, action: "confirm_same" })}>
                    <Link2 size={16} /> Same person
                  </button>
                  <button type="button" disabled={busy} onClick={() => void submitReview({ ...body, action: "mark_different" })}>
                    <Unlink2 size={16} /> Different
                  </button>
                  <button type="button" disabled={busy} onClick={() => void submitReview({ ...body, action: "unknown" })}>
                    <CircleHelp size={16} /> Unknown
                  </button>
                  <button type="button" className="danger" disabled={busy} onClick={() => void submitReview({ ...body, action: "reject", side: "both" })}>
                    <Trash2 size={16} /> Reject crops
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      )}

      <h3>Unreviewed tracks</h3>
      {!tracks.length ? (
        <p className="settings-help">No auto-labeled track galleries waiting for attention.</p>
      ) : (
        <div className="reid-track-list">
          {tracks.map((track) => (
            <article className="reid-track-card" key={`${track.event_id}-${track.track_id}`}>
              <header>
                <Images size={16} />
                <strong>{track.camera_id}</strong>
                <span>event {track.event_id} · track {track.track_id} · {track.sample_count} crops</span>
              </header>
              <SampleStrip samples={track.samples} title={track.person_id ? `person_${String(track.person_id).padStart(6, "0")}` : "anonymous"} />
              <div className="reid-pair-actions">
                <button
                  type="button"
                  className="danger"
                  disabled={busy}
                  onClick={() => void submitReview({
                    action: "reject",
                    left_event_id: track.event_id,
                    left_track_id: track.track_id,
                    right_event_id: track.event_id,
                    right_track_id: track.track_id,
                    side: "left",
                  })}
                >
                  <Trash2 size={16} /> Reject track crops
                </button>
              </div>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
