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

test('an open clip holds motion until end of stream, then a long event still ends', async t => {
  let now = 10_000;
  t.mock.method(Date, 'now', () => now);
  const recording = ready();
  recording.setMotion(true);
  const stream = recording.handleRecordingStreamRequest(1);
  try {
    assert.equal((await stream.next()).value!.isLast, false);
    recording.setMotion(false);
    assert.equal(recording.reportingMotion(), true);
    const early = await stream.next();
    assert.equal(early.value!.isLast, false);
    assert.equal(recording.reportingMotion(), true);
    now += 5000;
    recording.ring.append(Buffer.from('tail'), 4, 4);
    assert.equal((await stream.next()).value!.isLast, true);
    assert.equal((await stream.next()).done, true);
    assert.equal(recording.reportingMotion(), false);
    assert.equal(recording.counters.completed, 1);
  } finally { await stream.return(undefined); await recording.close(); }

  now = 20_000;
  const limited = ready();
  limited.setMotion(true);
  limited.ring.append(Buffer.from('later'), 4, 4);
  const long = limited.handleRecordingStreamRequest(2);
  try {
    assert.equal((await long.next()).value!.isLast, false);
    now += 60_000;
    assert.equal((await long.next()).value!.isLast, true);
    assert.equal(limited.counters.completed, 1);
  } finally { await long.return(undefined); await limited.close(); }
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
