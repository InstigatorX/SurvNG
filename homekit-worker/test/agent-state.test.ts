import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { Writable } from 'node:stream';
import { createServer } from 'node:http';
import { AgentFeeds, changedAssignments } from '../src/agent-state.js';

const child = () => spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)']);

test('an ended or rejected upload stops its encoder and waits for closure', async () => {
  const feeds = new AgentFeeds();
  const process = child();
  const closed = once(process, 'close');
  feeds.track('camera:1', process, Promise.resolve());
  await closed;
  assert.equal(feeds.has('camera:1'), false);
  await feeds.stopAll();
});

test('a late upload failure cannot stop a replacement encoder', async () => {
  const feeds = new AgentFeeds();
  let reject!: (reason: Error) => void;
  const upload = new Promise((_resolve, no) => { reject = no; });
  feeds.track('camera:1', child(), upload);
  await feeds.stop('camera:1');
  const replacement = child();
  feeds.track('camera:1', replacement, new Promise(() => {}));
  try {
    reject(new Error('old network request'));
    await new Promise(resolve => setTimeout(resolve, 30));
    assert.equal(feeds.has('camera:1'), true);
    assert.equal(replacement.exitCode, null);
  } finally { await feeds.stopAll(); }
});

test('a failed FFmpeg spawn is handled without an uncaught error', async () => {
  const feeds = new AgentFeeds();
  feeds.track('camera:1', spawn('/missing-survng-review-ffmpeg'), new Promise(() => {}));
  await new Promise(resolve => setTimeout(resolve, 30));
  assert.equal(feeds.has('camera:1'), false);
});

test('source, audio and software-permission changes invalidate an assignment', () => {
  const previous = new Map<string, { id: string; url: string; audio: boolean; allowSoftwareEncoding: boolean }>();
  const camera = { id: 'gate', url: 'rtsp://first', audio: false, allowSoftwareEncoding: true };
  assert.deepEqual(changedAssignments(previous, [camera]), []);
  assert.deepEqual(changedAssignments(previous, [{ ...camera, url: 'rtsp://second' }]), ['gate']);
  assert.deepEqual(changedAssignments(previous, [{ ...camera, audio: true }]), ['gate']);
  assert.deepEqual(changedAssignments(previous, [{ ...camera, allowSoftwareEncoding: false }]), ['gate']);
  assert.deepEqual(changedAssignments(previous, []), ['gate']);
});

test('a stalled encoder is killed even if it ignores SIGTERM', { timeout: 5000 }, async () => {
  const feeds = new AgentFeeds();
  const stubborn = spawn(process.execPath, ['-e', 'process.on("SIGTERM", () => {}); console.log("ready"); setInterval(() => {}, 1000)']);
  await once(stubborn.stdout!, 'data');
  feeds.track('camera:1', stubborn, new Promise(() => {}));
  await feeds.stopAll();
  assert.equal(stubborn.signalCode, 'SIGKILL');
  assert.deepEqual(feeds.keys(), []);
});

test('stopping a backpressured upload closes its pipes instead of waiting forever', { timeout: 5000 }, async () => {
  const feeds = new AgentFeeds();
  const producer = spawn(process.execPath, ['-e', 'process.stdout.write(Buffer.alloc(8*1024*1024)); setInterval(() => {}, 1000)']);
  producer.stderr!.resume();
  let blocked!: () => void;
  const ready = new Promise<void>(resolve => { blocked = resolve; });
  const sink = new Writable({ highWaterMark: 1, write(_chunk, _encoding, _done) { blocked(); } });
  producer.stdout!.pipe(sink);
  await ready;
  let cancelled = 0;
  feeds.track('camera:1', producer, new Promise(() => {}), () => { cancelled++; });
  let forcedDrain = false;
  const rescue = setTimeout(() => {
    forcedDrain = true; producer.stdout!.unpipe(sink); producer.stdout!.resume();
  }, 1000);
  try {
    await feeds.stopAll();
    assert.equal(forcedDrain, false, 'waiting for child close must not require resuming its blocked sink');
    assert.equal(cancelled, 1);
    assert.equal(producer.stdout!.destroyed, true);
    assert.deepEqual(feeds.keys(), []);
  } finally { clearTimeout(rescue); sink.destroy(); await feeds.stopAll(); }
});

test('a producer that exits also cancels its outstanding HTTP upload', async () => {
  const feeds = new AgentFeeds();
  const producer = spawn(process.execPath, ['-e', 'process.exit(0)']);
  let cancelled = 0;
  feeds.track('camera:1', producer, new Promise(() => {}), () => { cancelled++; });
  await once(producer, 'close');
  assert.equal(cancelled, 1);
  assert.deepEqual(feeds.keys(), []);
});

test('stopping a real stalled HTTP upload cancels the request and drains its child', { timeout: 5000 }, async () => {
  const server = createServer(request => request.pause());
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const feeds = new AgentFeeds();
  const producer = spawn(process.execPath, ['-e', 'process.stdout.write(Buffer.alloc(8*1024*1024)); setInterval(() => {}, 1000)']);
  producer.stderr!.resume();
  const abort = new AbortController();
  const arrived = once(server, 'request');
  const port = (server.address() as { port: number }).port;
  const upload = fetch(`http://127.0.0.1:${port}`, { method: 'POST', body: producer.stdout! as unknown as BodyInit,
    duplex: 'half', signal: abort.signal } as RequestInit);
  feeds.track('camera:1', producer, upload, () => abort.abort());
  let rescue: NodeJS.Timeout | undefined;
  try {
    await arrived;
    await new Promise(resolve => setTimeout(resolve, 30));
    let forced = false;
    rescue = setTimeout(() => { forced = true; producer.stdout!.destroy(); }, 1000);
    await feeds.stopAll();
    assert.equal(forced, false);
    assert.equal(abort.signal.aborted, true);
    assert.equal(producer.stdout!.destroyed, true);
  } finally {
    clearTimeout(rescue); abort.abort(); producer.stdout!.destroy();
    await feeds.stopAll(); server.closeAllConnections();
    await new Promise<void>(resolve => server.close(() => resolve()));
  }
});

test('background qualification is bounded, cancellable and drained before reuse', async () => {
  const { AgentJobs } = await import('../src/agent-state.js');
  const jobs = new AgentJobs(1);
  let entered = false;
  let cancelled = false;
  assert.equal(jobs.start('first', async signal => {
    entered = true;
    await new Promise<void>(resolve => signal.addEventListener('abort', () => { cancelled = true; resolve(); }, { once: true }));
  }, () => assert.fail('intentional cancellation is not a failure')), true);
  await Promise.resolve();
  assert.equal(entered, true);
  assert.equal(jobs.start('second', async () => {}, () => {}), false);
  await jobs.stopAll();
  assert.equal(cancelled, true);
  assert.equal(jobs.start('second', async () => {}, () => {}), true);
  await jobs.stopAll();
  assert.deepEqual(jobs.keys(), []);
});
