# Camera workspace experiment

Branch: `experiment/v1.2-camera-workspace`, based on `v1.2` at `a0a2d80`.

This experiment brings four concepts from the Velador camera workspace demo into SurvNG's existing Live and Timeline workspaces.

- **1 — Weighted camera mosaic.** Focus mode gives one camera a larger tile while fitting the remaining cameras around it. Use **Make primary** on a tile to change the focus; the selection survives reloads. Automatic and Custom layouts remain available. Custom supports direct move and resize controls. Focus and Custom use the desktop layout; phones retain the existing primary-camera layout.
- **2 — Contextual controls.** Hover or focus a tile to reveal Make primary, Live view, and Review footage. Touch layouts show these actions directly. Custom also exposes Move. Review footage carries the camera into Timeline. Recording and detection controls remain in the camera menu.
- **4 — Playback to clip.** Select a camera in Timeline and open Export. Seek to the desired points and use **Set start here** and **Set end here**, or adjust the existing selection handles. Start/End jump to the boundaries. **Preview clip** plays the selection and pauses at its end; Stop preview pauses immediately. The existing export job and download flow follows the selection. A clip must be at least one second and remain inside the displayed timeline bounds.
- **5 — Persistent health.** Live and Timeline show actual versus expected recording cameras, free storage, and an attention indicator. Details identify missing main/substreams, deliberately paused cameras, and unavailable status. Storage uses the configured cleanup/emergency thresholds. Status older than 90 seconds is treated as stale. The panel reuses shared snapshots and adds no network polling.

## Run the isolated preview

Requires Node, npm, and FFmpeg. From the repository:

```sh
cd frontend
npm ci
npm run preview:workspace
```

Open `http://127.0.0.1:5182/__preview`. Use its Live and Timeline links, or the Phone links for a 390 × 844 layout preview. `FFMPEG_PATH` can specify the FFmpeg executable; `PREVIEW_PORT` can choose another port.

The server binds to localhost and generates six synthetic camera scenes plus ten minutes of test-pattern footage. It uses temporary files, in-memory export jobs, and no real camera credentials or service connection. Stop it with Ctrl+C to remove the temporary media.

The control page switches between healthy, recording-failure, low-storage, and unavailable scenarios across all preview tabs. In unavailable mode, reload the app to see the initial error state, or leave it open for the existing polling/freshness checks to expire. Restore healthy afterward.

## Review checklist

1. Choose Focus, make another camera primary, then reload. Check that all cameras remain visible and the selection persists. Switch to Custom; move and resize a tile with pointer controls or Enter, arrow keys, Enter. Escape cancels an adjustment.
2. Reveal tile actions with pointer hover or keyboard focus. Review footage should open Timeline for that camera. On a phone-sized viewport, confirm the primary-camera selection and actions remain usable.
3. Use the control page's Timeline link to enter the generated recording window. Open Export, set a short start/end range, preview through the end, then try Stop preview. Submit a named export and check the queued-to-completed controls.
4. Trigger recording-failure and confirm 5/6 recording with Side gate missing its main stream. Trigger low-storage and confirm a critical storage warning. Trigger unavailable and confirm that old data does not appear healthy. Open/close Details by keyboard and check the panel at a phone width.

## Validation and limits

Run `npm test` for the frontend unit suite and `npm run build` for a production build. Focused coverage includes mosaic bounds/overlap and focus fallback, clip boundary validation, and health classification/freshness.

Browser checks use real media playback with generated footage. The preview's export endpoint simulates job progress and returns the full sample video; it does **not** encode the selected interval. Real camera transport, recorder recovery, exact exported clip contents, and backend export failures still require a trial against a running SurvNG instance. No capture, storage, or export backend behavior is changed by this branch.
