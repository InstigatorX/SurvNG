import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import type { ChildProcess } from 'node:child_process';
import type { CameraRecordingConfiguration } from '@homebridge/hap-nodejs';
import { FragmentRing, LabRecording } from '../src/fragments.js';
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
  const stream = recording.handleRecordingStreamRequest(1);
  try {
    assert.equal((await stream.next()).value!.isLast, false);
    assert.equal((await stream.next()).value!.isLast, false);
    now += 1000;
    recording.setMotion(false);
    now += 3000;
    recording.ring.append(Buffer.from('tail'), 4, 4);
    assert.equal((await stream.next()).value!.isLast, true);
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
    assert.equal(recording.reportingMotion(), true);
    assert.equal(recording.status().download, 'awaiting_ack');
    await assert.rejects(recording.handleRecordingStreamRequest(3).next(), /download already active/);
    assert.equal(recording.status().openRejected.busy, 1);
    assert.equal(recording.counters.completed, 1);
    assert.equal(recording.counters.acknowledged, 0);
    recording.setMotion(true);
    recording.acknowledgeStream(1);
    assert.equal(recording.counters.acknowledged, 1);
    assert.equal(recording.status().download, 'idle');
    assert.equal(recording.reportingMotion(), true);
    recording.setMotion(false);
    assert.equal(recording.reportingMotion(), false);
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

test('a hub close before acknowledgement releases the sensor without accepting the clip', async t => {
  let now = 10_000;
  t.mock.method(Date, 'now', () => now);
  const recording = ready();
  recording.setMotion(true);
  recording.setMotion(false);
  const stream = recording.handleRecordingStreamRequest(1);
  try {
    await stream.next();
    assert.equal((await stream.next()).value!.isLast, false);
    now += 5000;
    recording.ring.append(Buffer.from('tail'), 4, 4);
    assert.equal((await stream.next()).value!.isLast, true);
    await stream.next();
    assert.equal(recording.reportingMotion(), true);
    recording.closeRecordingStream(1);
    assert.equal(recording.reportingMotion(), false);
    assert.equal(recording.counters.acknowledged, 0);
    assert.equal(recording.counters.closed, 0);
  } finally { await stream.return(undefined); await recording.close(); }
});

test('a late recording open still records a post-roll', async t => {
  let now = 10_000;
  t.mock.method(Date, 'now', () => now);
  const recording = ready();
  recording.setMotion(true);
  recording.setMotion(false);
  now += 5000;
  const stream = recording.handleRecordingStreamRequest(1);
  try {
    assert.equal((await stream.next()).value!.isLast, false);
    assert.equal((await stream.next()).value!.isLast, false);
    now += 5000;
    recording.ring.append(Buffer.from('tail'), 4, 4);
    assert.equal((await stream.next()).value!.isLast, true);
    assert.equal(recording.counters.completed, 1);
  } finally { await stream.return(undefined); await recording.close(); }
});

test('producer loss fails an open recording without resending its last fragment', async () => {
  const recording = ready();
  recording.setMotion(true);
  const stream = recording.handleRecordingStreamRequest(1);
  try {
    await stream.next();
    assert.deepEqual((await stream.next()).value, { data: Buffer.from('first'), isLast: false });
    recording.setMotion(false);
    const waiting = stream.next();
    recording.ring.close();
    await assert.rejects(waiting, /fragment source closed/);
    assert.equal(recording.counters.failed, 1);
    assert.equal(recording.counters.completed, 0);
    assert.equal(recording.counters.acknowledged, 0);
    assert.equal(recording.reportingMotion(), false);
  } finally { await stream.return(undefined); await recording.close(); }
});

test('recording status distinguishes a rejected open from an idle download', async () => {
  const recording = ready();
  recording.ring = new FragmentRing();
  try {
    await assert.rejects(recording.handleRecordingStreamRequest(10).next(), /prebuffer not ready/);
    assert.equal(recording.status().openRequests, 1);
    assert.equal(recording.status().openRejected.prebuffer, 1);
    assert.equal(recording.counters.streams, 0);
    assert.equal(recording.counters.failed, 0);
  } finally { await recording.close(); }
});

test('recording activation waits for producer admission and retires it when disabled', async () => {
  const child = Object.assign(new EventEmitter(), {
    exitCode: null, signalCode: null, stdio: [], stdout: new EventEmitter(),
    kill: () => { queueMicrotask(() => child.emit('close')); return true; },
  });
  let beginAdmission!: () => void;
  const admissionBegan = new Promise<void>(resolve => { beginAdmission = resolve; });
  let admit!: (producer: ChildProcess) => void;
  const admitted = new Promise<ChildProcess>(resolve => { admit = resolve; });
  const media = new SyntheticMedia('/unused-review-fixture');
  const recording = new LabRecording({
    children: media.children,
    start: () => { beginAdmission(); return admitted; },
    input: tier => media.input(tier),
    video: tier => media.video(tier),
    recording: (tier, audio) => media.recording(tier, audio),
    snapshot: () => media.snapshot(),
    stop: () => media.stop(),
  }, () => true);
  recording.updateRecordingConfiguration({ videoCodec: { resolution: [1920, 1080, 30] } } as CameraRecordingConfiguration);
  recording.updateRecordingActive(true);
  await admissionBegan;
  const duringAdmission = recording.status();
  const closed = recording.close();
  admit(child as unknown as ChildProcess);
  await closed;
  assert.equal(duringAdmission.requested, true);
  assert.equal(duringAdmission.active, false);
  assert.equal(recording.status().active, false);
  assert.equal(recording.counters.producerFailures, 0);
  assert.equal(child.listenerCount('close'), 0);
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
