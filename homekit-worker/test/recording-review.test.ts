import { test } from 'node:test';
import assert from 'node:assert/strict';
import { LabRecording } from '../src/fragments.js';
import { SyntheticMedia } from '../src/media.js';

function ready(allowed = () => true): LabRecording {
  const recording = new LabRecording(new SyntheticMedia('/unused-review-fixture'), allowed);
  Object.assign(recording, { active: true, init: Buffer.from('init') });
  recording.ring.append(Buffer.from('first'), 0, 4);
  return recording;
}

test('repeated false motion heartbeats do not extend recording post-roll', async t => {
  let now = 10000;
  t.mock.method(Date, 'now', () => now);
  const recording = ready();
  recording.setMotion(true);
  recording.setMotion(false);
  now += 5000;
  recording.setMotion(false);
  const stream = recording.handleRecordingStreamRequest(1);
  try {
    assert.equal((await stream.next()).value!.isLast, false);
    assert.equal((await stream.next()).value!.isLast, true);
    assert.equal((await stream.next()).done, true);
  } finally { await stream.return(undefined); await recording.close(); }
});

test('privacy changed while a recording read waits prevents the next fragment', async () => {
  let allowed = true;
  const recording = ready(() => allowed);
  recording.setMotion(true);
  const stream = recording.handleRecordingStreamRequest(1);
  try {
    await stream.next();
    await stream.next();
    const waiting = stream.next();
    allowed = false;
    recording.ring.append(Buffer.from('private'), 4, 4);
    await assert.rejects(waiting, /privacy|unavailable/);
  } finally { await stream.return(undefined); await recording.close(); }
});
