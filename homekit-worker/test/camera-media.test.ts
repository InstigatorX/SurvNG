import { test } from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { mkdtemp, readFile, rm, writeFile, readdir, lstat, chmod } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { execFile, type ChildProcess } from 'node:child_process';
import { promisify } from 'node:util';
import { createServer } from 'node:http';
import { CameraMedia, encoderCommand, remoteCopyCommand, supportsReadrateCatchup } from '../src/camera-media.js';
import { RtpFeed, stopChild, tiers } from '../src/media.js';
import { FragmentParser, FragmentRing, videoClock, decodeTime, fragmentDuration } from '../src/fragments.js';
import { HevcAccessUnitAssembler } from '../src/vendor/sframeRtp.js';

async function until(predicate: () => boolean | Promise<boolean>, timeout = 15000): Promise<void> {
  const end = Date.now() + timeout;
  while (!await predicate()) {
    if (Date.now() > end) throw new Error('condition timed out');
    await new Promise(resolve => setTimeout(resolve, 25));
  }
}

class FileCameraMedia extends CameraMedia {
  // A file-backed test camera must advance in real time like an RTSP source.
  protected override directInput(): string[] { return ['-re', ...super.directInput()]; }
}

test('a remote feed is copied and a local encoder keeps the source', () => {
  const remote = remoteCopyCommand('/srv/homekit/feeds/gate/1.sock').join(' ');
  assert.equal(remote.includes('unix:///srv/homekit/feeds/gate/1.sock'), true);
  assert.equal(remote.includes('+genpts+igndts'), true);
  assert.equal(remote.includes('-analyzeduration 0'), true);
  assert.equal(remote.includes('-c:v copy'), true);
  assert.equal(remote.includes('-map 0:a:0 -c:a copy'), true);
  assert.equal(remote.includes('pad='), false);
  const local = encoderCommand('rtsp://camera/live', 'tcp', false, tiers[0], { backend: 'software' }).join(' ');
  assert.equal(local.includes('rtsp://camera/live'), true);
  assert.equal(local.includes('libx265'), true);
});

test('relay pacing uses only the optional catch-up flag reported by its FFmpeg', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-ffmpeg-help-'));
  try {
    for (const supported of [false, true]) {
      const ffmpeg = join(root, `ffmpeg-${supported}`);
      const option = supported ? '-readrate_catchup <speed>' : '-readrate_catchup_other <speed>';
      await writeFile(ffmpeg, `#!${process.execPath}
if (process.argv.slice(2).join(' ') !== '-hide_banner -h long') process.exit(1);
console.log('-readrate_initial_burst <seconds>\\n${option}');
`, { mode: 0o700 });
      const readrateCatchup = await supportsReadrateCatchup(ffmpeg);
      assert.equal(readrateCatchup, supported);
      const media = new CameraMedia('unused', true, ffmpeg, 'tcp', async () => {},
        () => true, root, { backend: 'software' }, tiers, '', readrateCatchup);
      for (const args of [media.input(tiers[0]), media.recording(tiers[1], true)]) {
        assert.equal(args[args.indexOf('-analyzeduration') + 1], '0');
        assert.ok(args.includes('-re'));
        assert.equal(args[args.indexOf('-readrate_initial_burst') + 1], '0.001');
        assert.equal(args.includes('-readrate_catchup'), supported);
        if (supported) assert.equal(args[args.indexOf('-readrate_catchup') + 1], '1');
      }
    }
    await assert.rejects(supportsReadrateCatchup(join(root, 'missing')), { code: 'ENOENT' });
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('RTSP sources use the demuxer timeout while HTTP uses the protocol timeout', () => {
  class Inputs extends CameraMedia { args(): string[] { return this.directInput(); } }
  for (const scheme of ['rtsp', 'rtsps', 'RTSP']) {
    const args = new Inputs(`${scheme}://not-opened/camera`, true, 'ffmpeg', 'tcp', async () => {}).args();
    assert.ok(args.includes('-timeout'));
    assert.ok(args.includes('-rtsp_transport'));
    assert.equal(args.includes('-rw_timeout'), false);
  }
  const http = new Inputs('http://not-opened/camera', true, 'ffmpeg', 'tcp', async () => {}).args();
  assert.ok(http.includes('-rw_timeout'));
  assert.equal(http.includes('-timeout'), false);
  assert.equal(http.includes('-rtsp_transport'), false);
});

test('four-second fragment coverage and preroll use media durations', () => {
  const ring = new FragmentRing(1024, 12);
  ring.append(Buffer.alloc(10), 0, 4);
  assert.equal(ring.coverage, 4); assert.equal(ring.cursor(4), 1);
  ring.append(Buffer.alloc(10), 4, 4);
  ring.append(Buffer.alloc(10), 8, 4);
  assert.equal(ring.cursor(5), 2); // include the boundary fragment
  ring.append(Buffer.alloc(10), 12, 4);
  assert.equal(ring.coverage, 12); assert.equal(ring.fragments[0].start, 4);
  assert.throws(() => ring.append(Buffer.alloc(1), 20, 4), /discontinuity/);
});

test('privacy during media admission prevents a process from starting', async () => {
  let allowed = true;
  const messages: string[] = [];
  const media = new CameraMedia('rtsp://not-opened/camera', false, 'not-executed', 'tcp', async (method) => {
    messages.push(method);
    if (method === 'acquire') allowed = false;
  }, () => allowed);
  await assert.rejects(media.start(['-version']), /unavailable/);
  assert.deepEqual(messages, ['acquire', 'release']);
  assert.equal(media.children.size, 0);
  await media.stop();
});

test('an intentional encoder stop does not report FFmpeg exit 255 as a failure', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-camera-stop-'));
  const executable = join(root, 'encoder');
  await writeFile(executable, `#!${process.execPath}\nprocess.on('SIGTERM', () => process.exit(255));\nsetInterval(() => {}, 1000);\nprocess.stdout.write('ready');\n`);
  await chmod(executable, 0o700);
  const media = new CameraMedia('rtsp://not-opened/camera', false, executable, 'tcp', async () => {});
  try {
    const child = await media.start([]);
    await once(child.stdout!, 'data');
    await media.stop();
    assert.equal(child.exitCode, 255);
    assert.equal(media.counters.failures, 0);
    assert.equal(media.children.size, 0);
  } finally { await media.stop(); await rm(root, { recursive: true, force: true }); }
});

test('stopping a feed waits for an outstanding media admission', async () => {
  let grant!: () => void;
  const ready = new Promise<void>(resolve => { grant = resolve; });
  let acquired!: () => void;
  const requested = new Promise<void>(resolve => { acquired = resolve; });
  const messages: string[] = [];
  const media = new CameraMedia('rtsp://not-opened/camera', false, process.execPath, 'tcp', async method => {
    messages.push(method);
    if (method === 'acquire') { acquired(); await ready; }
  });
  // Use Node's --version for a harmless short-lived admitted child.
  const realStart = media.start.bind(media);
  media.start = () => realStart(['--version']);
  const feed = new RtpFeed(media, tiers[2]);
  const starting = feed.start(() => {}, () => {}, () => {});
  await requested;
  const stopping = feed.stop();
  grant();
  await Promise.all([starting, stopping]);
  await media.stop();
  assert.equal(media.children.size, 0);
  assert.deepEqual(messages, ['acquire', 'started', 'release']);
});

test('shared relay cleanup survives a delayed started acknowledgement', { timeout: 15000 }, async t => {
  const root = await mkdtemp(join(tmpdir(), 'survng-relay-ack-'));
  const executable = join(root, 'encoder');
  await writeFile(executable, `#!${process.execPath}
const net = require('node:net');
const input = process.argv.find(arg => arg.startsWith('unix://'));
if (input) {
  const client = net.connect(input.slice(7));
  client.once('data', () => {
    if (process.argv.includes('early-exit')) {
      process.stderr.write('invalid relay output');
      process.exitCode = 7;
      client.destroy();
    } else process.stdout.write('ready');
  });
} else {
  setInterval(() => process.stdout.write('packet'), 20);
}
`, { mode: 0o700 });
  try {
    for (const outcome of ['success after close', 'rejection after close', 'rejection while running']) {
      await t.test(outcome, async () => {
        const leases = new Set<string>();
        const releases: string[] = [];
        let grant!: () => void, deny!: (error: Error) => void, requested!: () => void;
        const ack = new Promise<void>((resolve, reject) => { grant = resolve; deny = reject; });
        const ackRequested = new Promise<void>(resolve => { requested = resolve; });
        let delayed = '';
        let delayNextRelay = false;
        const media = new CameraMedia('unused', true, executable, 'tcp', async (method, token, detail) => {
          if (method === 'acquire') {
            leases.add(token);
            if (detail?.purpose === 'relay' && delayNextRelay) delayed = token;
          } else if (method === 'started' && token === delayed) {
            requested();
            await ack;
          } else if (method === 'release') {
            releases.push(token);
            leases.delete(token);
          }
        }, () => true, root);
        try {
          const input = media.input(tiers[0]);
          const sibling = await media.start(input);
          await once(sibling.stdout!, 'data');
          delayNextRelay = true;
          const exits = outcome !== 'rejection while running';
          const starting = media.start(exits ? [...input, 'early-exit'] : input);
          await ackRequested;
          if (exits) await until(() => releases.includes(delayed));
          const drained = async () => media.children.size === 0 && leases.size === 0
            && !(await readdir(root)).some(name => name.endsWith('.sock'));
          if (outcome === 'success after close') {
            assert.equal(media.children.size, 2, 'one sibling relay and its producer remain');
            await stopChild(sibling);
            await until(drained, 2000); // The pending ACK cannot keep the producer or its socket alive.
            grant();
          } else deny(new Error('acknowledgement rejected'));
          await assert.rejects(starting, /exited|ownership unavailable/);
          if (exits) assert.equal(media.lastError, 'media_error');
          if (outcome !== 'success after close') {
            assert.equal(sibling.exitCode, null);
            assert.equal(media.children.size, 2, 'the failure releases only its own reference');
            assert.equal((await readdir(root)).filter(name => name.endsWith('.sock')).length, 1);
            await stopChild(sibling);
            await until(drained, 2000);
          }
          assert.equal(releases.length, 3, 'each relay and producer lease released once');
        } finally { grant(); await media.stop(); }
      });
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('a relay spawn error releases its shared source and admission', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-relay-spawn-'));
  const messages: string[] = [];
  const media = new CameraMedia('unused', true, join(root, 'missing'), 'tcp',
    async method => { messages.push(method); }, () => true, root);
  try {
    await assert.rejects(media.start(media.input(tiers[0])), /unavailable/);
    await until(() => messages.includes('release'));
    assert.deepEqual(messages, ['acquire', 'release']);
    assert.deepEqual(await readdir(root), []);
    assert.equal(media.children.size, 0);
  } finally { await media.stop(); await rm(root, { recursive: true, force: true }); }
});

test('camera HTTP input is normalized to HEVC/AAC fMP4 with a removable audio track', {
  skip: process.env.HKSV_MEDIA_TESTS !== '1', timeout: 60000,
}, async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-camera-media-'));
  const encoding = process.env.HKSV_QSV_TESTS === '1' ? { backend: 'qsv' as const, device: '/dev/dri/renderD128' } : { backend: 'software' as const };
  const execute = promisify(execFile);
  const ffmpeg = process.env.HKSV_FFMPEG || 'ffmpeg';
  const ffprobe = join(dirname(ffmpeg), 'ffprobe');
  const readrateCatchup = await supportsReadrateCatchup(ffmpeg);
  const input = join(root, 'camera.mp4');
  const admitted = new Set<string>();
  let encoderAcquires = 0;
  let media: CameraMedia | undefined;
  const server = createServer();
  try {
    await execute(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=15',
      '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=32000', '-t', '8', '-c:v', 'libx264', '-preset', 'ultrafast',
      '-c:a', 'aac', '-movflags', '+faststart', '-y', input]);
    const fixture = await readFile(input);
    server.on('request', (request, response) => {
      const range = /^bytes=(\d+)-(\d*)/.exec(request.headers.range || '');
      const start = range ? Number(range[1]) : 0;
      const end = range?.[2] ? Number(range[2]) : fixture.length - 1;
      response.writeHead(range ? 206 : 200, { 'Content-Type': 'video/mp4', 'Accept-Ranges': 'bytes',
        'Content-Length': end - start + 1, ...(range ? { 'Content-Range': `bytes ${start}-${end}/${fixture.length}` } : {}) });
      response.end(fixture.subarray(start, end + 1));
    });
    server.listen(0, '127.0.0.1'); await once(server, 'listening');
    const address = server.address() as { port: number };
    media = new FileCameraMedia(`http://127.0.0.1:${address.port}/camera.mp4`, true, ffmpeg, 'tcp', async (method, token, detail) => {
      if (method === 'acquire' && detail?.purpose === 'encoder') encoderAcquires++;
      if (method === 'acquire') admitted.add(token); else if (method === 'release') admitted.delete(token);
    }, () => true, root, encoding, tiers, '', readrateCatchup);
    for (const audio of [true, false]) {
      const child: ChildProcess = await media.start([...media.recording(tiers[2], audio), '-tag:v', 'hvc1',
        '-movflags', '+frag_keyframe+empty_moov+default_base_moof', '-f', 'mp4', 'pipe:1']);
      const chunks: Buffer[] = [];
      child.stdout!.on('data', (chunk: Buffer) => chunks.push(chunk));
      if (audio) {
        let frames = 0, audioPackets = 0;
        let firstFrame = 0, lastFrame = 0;
        const assembler = new HevcAccessUnitAssembler();
        const annexB: Buffer[] = [];
        const feed = new RtpFeed(media, tiers[2]);
        try {
          await feed.start(packet => {
            const frame = assembler.push(packet);
            if (!frame) return;
            frames++;
            lastFrame = performance.now(); if (frames === 1) firstFrame = lastFrame;
            for (let offset = 0; offset < frame.data.length;) {
              const length = frame.data.readUInt32BE(offset);
              annexB.push(Buffer.from([0, 0, 0, 1]), frame.data.subarray(offset + 4, offset + 4 + length));
              offset += 4 + length;
            }
          }, () => audioPackets++, () => {});
          const sockets = (await readdir(root)).filter(name => name.endsWith('.sock'));
          assert.equal(sockets.length, 1);
          assert.equal((await lstat(join(root, sockets[0]))).mode & 0o777, 0o600);
          await until(() => frames >= 60 && audioPackets >= 10);
          const mediaSpan = (frames - 1) / tiers[2].frameRate * 1000;
          // Allow codec startup latency, but reject demuxer's multi-second dump
          // and its default half-second readrate burst over a four-second window.
          assert.ok(lastFrame - firstFrame >= mediaSpan - 300,
            'startup probe must not dump buffered video faster than its media clock');
          assert.equal(encoderAcquires, 1, 'live and recording share one HEVC producer');
          const snapshot = await media.snapshot();
          assert.equal(snapshot.readUInt16BE(0), 0xffd8);
          assert.equal(encoderAcquires, 1, 'snapshot does not start another HEVC encoder');
        } finally { await feed.stop(); }
        const liveFile = join(root, 'camera-live.hevc');
        await writeFile(liveFile, Buffer.concat(annexB));
        const decoded = await execute(ffmpeg, ['-v', 'error', '-i', liveFile, '-f', 'null', '-']);
        assert.equal(decoded.stderr, '');
      }
      const code: number = (await once(child, 'close'))[0];
      assert.equal(code, 0, media.lastError);
      await until(() => admitted.size === 0, 3000);
      assert.equal(admitted.size, 0);
      const output = Buffer.concat(chunks);
      const file = join(root, `recording-${audio}.mp4`); await writeFile(file, output);
      const probe = JSON.parse((await execute(ffprobe, ['-v', 'error', '-show_streams', '-of', 'json', file])).stdout);
      const video = probe.streams.find((stream: { codec_type: string }) => stream.codec_type === 'video');
      assert.equal(video.codec_name, 'hevc'); assert.equal(video.width, 640); assert.equal(video.height, 360);
      assert.equal(probe.streams.filter((stream: { codec_type: string }) => stream.codec_type === 'audio').length, audio ? 1 : 0);
      let clock: { track: number; scale: number } | undefined;
      const ring = new FragmentRing();
      const parser = new FragmentParser(init => { clock = videoClock(init); }, fragment => {
        ring.append(fragment, Number(decodeTime(fragment, clock!.track)) / clock!.scale,
          fragmentDuration(fragment, clock!.track, clock!.scale));
      });
      parser.push(output);
      assert.ok(ring.coverage >= 7.9 && ring.coverage < 8.1);
      assert.ok(Math.abs(ring.fragments[0].duration - 4) < 0.1);
      assert.doesNotThrow(() => ring.cursor(4));
      await execute(ffmpeg, ['-v', 'error', '-i', file, '-f', 'null', '-']);
    }
    await media.stop();
    media = new FileCameraMedia(`http://127.0.0.1:${address.port}/camera.mp4`, false, ffmpeg, 'tcp', async () => {}, () => true, root, encoding, tiers, '', readrateCatchup);
    const silentChild: ChildProcess = await media.start([...media.recording(tiers[2], true), '-t', '4', '-tag:v', 'hvc1',
      '-movflags', '+frag_keyframe+empty_moov+default_base_moof', '-f', 'mp4', 'pipe:1']);
    const silentChunks: Buffer[] = [];
    silentChild.stdout!.on('data', (chunk: Buffer) => silentChunks.push(chunk));
    assert.equal((await once(silentChild, 'close'))[0], 0);
    const silentFile = join(root, 'camera-audio-disabled.mp4');
    await writeFile(silentFile, Buffer.concat(silentChunks));
    const pcm = await execute(ffmpeg, ['-v', 'error', '-i', silentFile, '-map', '0:a:0', '-f', 's16le', 'pipe:1'], { encoding: 'buffer', maxBuffer: 1024 * 1024 });
    assert.ok(pcm.stdout.length > 10000);
    assert.ok(pcm.stdout.every(byte => byte === 0), 'camera-audio-disabled produces silence');
  } finally {
    await media?.stop();
    await new Promise<void>(resolve => server.close(() => resolve()));
    await rm(root, { recursive: true, force: true });
  }
});

test('a remote snapshot removes its temporary JPEG and honors privacy', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-remote-snapshot-'));
  let allowed = true;
  let requests = 0;
  const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xd9]);
  const path = join(root, 'remote-snapshot.jpg');
  const media = new CameraMedia('unused', false, 'ffmpeg', 'tcp', async method => {
    assert.equal(method, 'snapshot'); requests++;
    await writeFile(path, jpeg, { mode: 0o600 });
  }, () => allowed, root, { backend: 'qsv', device: '/unused' }, tiers, '/unused-feed');
  try {
    const results = await Promise.all([media.snapshot(), media.snapshot()]);
    assert.deepEqual(results, [jpeg, jpeg]);
    assert.equal(requests, 1, 'concurrent thumbnails share one request through JPEG cleanup');
    await assert.rejects(readFile(path), { code: 'ENOENT' });
    allowed = false;
    await assert.rejects(media.snapshot(), /unavailable/);
    assert.equal(requests, 1);
  } finally { await media.stop(); await rm(root, { recursive: true, force: true }); }
});
