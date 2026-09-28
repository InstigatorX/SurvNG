import { useEffect, useState } from "react";
import { incidentEpisodeClips } from "../incidentScene.mjs";

export function useIncidentPlayback(incident) {
  const episodes = incidentEpisodeClips(incident);
  const [selection, setSelection] = useState(null);
  useEffect(() => setSelection(null), [incident?.incident_id || incident?.id]);
  const index = selection ? episodes.findIndex((episode) => episode.id === selection.episodeId) : -1;
  const episode = episodes[index] || null;
  function select(episodeId, continuous = false) {
    const selected = episodes.find((item) => item.id === episodeId && item.clip);
    // Pin this request's bounds; live refresh must not restart a playing clip.
    if (selected) setSelection({ episodeId, continuous, key: Date.now(), clip: selected.clip });
  }
  function playAll() {
    const first = episodes.find((item) => item.clip);
    if (first) select(first.id, true);
  }
  function next() {
    const following = episodes.slice(index + 1).find((item) => item.clip && (selection?.continuous || item.episode_id === episode?.episode_id));
    if (following) select(following.id, selection?.continuous ?? true);
    else setSelection(null);
  }
  const nextCamera = episodes.slice(index + 1).find((item) => item.clip && item.episode_id !== episode?.episode_id);
  return { episodes, episode, clip: episode ? selection.clip : null, selection, index, select, playAll, next,
    stop: () => setSelection(null),
    ended: next,
    nextEpisode: () => { if (nextCamera) select(nextCamera.id, selection?.continuous ?? true); },
    hasNext: Boolean(nextCamera),
  };
}
