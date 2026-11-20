import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { encodeVideo, deviceArgs, selectEncoding, qualifyEncoding, streamProfile, recordingProfile, selectCellularTier, EncodingUnavailableError } from '../src/encoding.js';
import { LowDelayProof } from '../src/low-delay.js';
import { tiers } from '../src/media.js';

test('QSV is preferred; software fallback requires permission and a successful probe', async () => {
  let softwareProbes = 0;
  const software = async () => { softwareProbes++; return true; };
  assert.equal((await selectEncoding(false, async () => true, software)).backend, 'qsv');
  assert.equal(softwareProbes, 0);
  await assert.rejects(selectEncoding(false, async () => false, software), EncodingUnavailableError);
  assert.equal(softwareProbes, 0);
  assert.deepEqual(await selectEncoding(true, async () => false, software), { backend: 'software', fallbackReason: 'qsv_unavailable' });
  await assert.rejects(selectEncoding(true, async () => false, async () => false), EncodingUnavailableError);
  await assert.rejects(selectEncoding(true, async () => { throw new Error('admission unavailable'); }, software), /admission unavailable/);
});

test('hardware qualification respects cancellation before acquiring resources', async () => {
  const abort = new AbortController(); abort.abort();
  let admitted = false;
  await assert.rejects(qualifyEncoding('unused', '/unused', async () => { admitted = true; }, abort.signal,
    { backend: 'qsv', device: '/unused' }));
  assert.equal(admitted, false);
});

test('stream geometry follows the camera aspect within each tier budget', () => {
  // Every tier, including the top one, fits the cellular relay envelope.
  for (const [width, height] of [[1920, 1080], [2560, 1920], [1080, 1920], [1440, 1920], [1080, 1080], [3840, 1080]]) {
    const profile = streamProfile(width, height);
    assert.equal(selectCellularTier(profile.tiers), profile.tiers[0]);
    for (const tier of profile.tiers) {
      assert.ok(tier.peakBitrate <= 800 && Math.max(tier.width, tier.height) <= 1280 && Math.min(tier.width, tier.height) <= 720);
    }
  }
  const landscape = streamProfile(896, 672);
  assert.deepEqual(landscape.tiers.map(tier => [tier.width, tier.height]), [[960, 720], [800, 600], [640, 480]]);
  const portrait = streamProfile(672, 896);
  assert.deepEqual(portrait.tiers.map(tier => [tier.width, tier.height]), [[720, 960], [600, 800], [480, 640]]);
  assert.deepEqual(streamProfile(1920, 1080).tiers.map(tier => [tier.width, tier.height]), [[1280, 720], [960, 540], [640, 360]]);
  assert.deepEqual(streamProfile(1920, 832).tiers.map(tier => [tier.width, tier.height]), [[1280, 554], [960, 416], [640, 278]]);
  assert.equal(encodeVideo(landscape.tiers[0], { backend: 'software' }).join(' ').includes('crop='), false);
  assert.equal(encodeVideo(landscape.tiers[0], { backend: 'qsv', device: '/dev/dri/renderD128' }).join(' ').includes('-fps_mode cfr'), true);
  assert.equal(encodeVideo(landscape.tiers[0], { backend: 'qsv', device: '/dev/dri/renderD128' }).join(' ').includes('dump_extra=freq=keyframe'), true);
});

test('temporal qualification rejects reordering, future references, missing headers and open GOPs', () => {
  const proof = (packets: string, headers = '') => {
    const value = new LowDelayProof();
    value.push('packets', Buffer.from(packets));
    value.push('headers', Buffer.from('sps_max_num_reorder_pics[0] 1 = 0\nnal_unit_type 10011 = 19\nnal_unit_type 10011 = 19\n' + headers));
    return value;
  };
  assert.equal(proof('0, 0, 0, 1\n0, 1, 1, 1\n').valid(2), true);
  assert.equal(proof('0, 0, 1, 1\n0, 1, 0, 1\n').valid(2), false);
  assert.equal(proof('0, 1, 1, 1\n0, 0, 0, 1\n').valid(2), false);
  assert.equal(proof('0, 0, 0, 1\n', 'sps_max_num_reorder_pics[0] 010 = 1\n').valid(1), false);
  assert.equal(proof('0, 0, 0, 1\n', 'num_positive_pics 010 = 1\n').valid(1), false);
  assert.equal(proof('0, 0, 0, 1\n', 'nal_unit_type 10101 = 21\n').valid(1), false);
  assert.equal(new LowDelayProof().valid(0), false);
});

test('wide camera encoding retains both edge markers at every tier', {
  skip: process.env.HKSV_MEDIA_TESTS !== '1', timeout: 30000,
}, async () => {
  const execute = promisify(execFile);
  const ffmpeg = process.env.HKSV_FFMPEG || 'ffmpeg';
  const root = await mkdtemp(join(tmpdir(), 'survng-wide-'));
  const width = 1920, height = 832;
  const pixels = Buffer.alloc(width * height * 3, 80);
  for (let y = 0; y < height; y++) for (let x = 0; x < 80; x++) {
    const left = (y * width + x) * 3;
    const right = (y * width + width - 1 - x) * 3;
    pixels.set([255, 0, 0], left); pixels.set([0, 0, 255], right);
  }
  const source = join(root, 'edges.rgb');
  try {
    await writeFile(source, pixels);
    const encodings = [{ backend: 'software' as const }, ...(process.env.HKSV_QSV_TESTS === '1'
      ? [{ backend: 'qsv' as const, device: '/dev/dri/renderD128' }] : [])];
    const cases = [...streamProfile(width, height).tiers.map(tier => ({ name: `native-${tier.identifier}`, tier })),
      { name: 'stale-16x9-profile', tier: tiers[0] }];
    for (const encoding of encodings) for (const { name, tier } of cases) {
      const output = join(root, `${encoding.backend}-${name}.mp4`);
      await execute(ffmpeg, ['-hide_banner', '-loglevel', 'error', ...deviceArgs(encoding),
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', `${width}x${height}`, '-framerate', '30', '-i', source,
        '-frames:v', '1', ...encodeVideo(tier, encoding), '-tag:v', 'hvc1', output], { timeout: 10000 });
      const { stdout } = await execute(ffmpeg, ['-v', 'error', '-i', output, '-frames:v', '1',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], { encoding: 'buffer', maxBuffer: 8 * 1024 * 1024, timeout: 10000 });
      assert.equal(stdout.length, tier.width * tier.height * 3);
      const sample = (fraction: number) => {
        const offset = (Math.floor(tier.height / 2) * tier.width + Math.floor(tier.width * fraction)) * 3;
        return [...stdout.subarray(offset, offset + 3)];
      };
      const [lr, lg, lb] = sample(0.02), [rr, rg, rb] = sample(0.98);
      assert.ok(lr > 180 && lg < 80 && lb < 80, `${encoding.backend} tier ${tier.identifier} lost the left edge`);
      assert.ok(rb > 180 && rr < 80 && rg < 80, `${encoding.backend} tier ${tier.identifier} lost the right edge`);
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('QSV produces decodable Main HEVC with periodic IDRs and zero temporal reordering at every tier', {
  skip: process.env.HKSV_QSV_TESTS !== '1', timeout: 60000,
}, async () => {
  const execute = promisify(execFile);
  const ffmpeg = process.env.HKSV_FFMPEG || 'ffmpeg';
  const encoding = { backend: 'qsv' as const, device: '/dev/dri/renderD128' };
  const root = await mkdtemp(join(tmpdir(), 'survng-qsv-'));
  try {
    assert.equal(await qualifyEncoding(ffmpeg, root, async () => {}, new AbortController().signal, encoding), true);
    for (const tier of tiers) {
      const file = join(root, `${tier.identifier}.mp4`);
      const count = tier.frameRate * 4 + 5;
      await execute(ffmpeg, ['-hide_banner', '-loglevel', 'error', ...deviceArgs(encoding),
        '-f', 'lavfi', '-i', `testsrc2=size=${tier.width}x${tier.height}:rate=${tier.frameRate}`, '-frames:v', String(count),
        ...encodeVideo(tier, encoding), '-tag:v', 'hvc1', '-f', 'mp4', file], { timeout: 20000 });
      const probe = JSON.parse((await execute('ffprobe', ['-v', 'error', '-show_streams', '-show_frames', '-of', 'json', file], { maxBuffer: 4 * 1024 * 1024 })).stdout);
      assert.equal(probe.streams[0].codec_name, 'hevc'); assert.equal(probe.streams[0].profile, 'Main');
      assert.equal(probe.streams[0].width, tier.width); assert.equal(probe.streams[0].height, tier.height);
      assert.equal(probe.frames.length, count);
      const proof = new LowDelayProof();
      const traced = await execute(ffmpeg, ['-hide_banner', '-loglevel', 'info', '-i', file, '-c:v', 'copy',
        '-bsf:v', 'hevc_mp4toannexb,trace_headers', '-f', 'framehash', 'pipe:1'], { maxBuffer: 8 * 1024 * 1024 });
      proof.push('packets', Buffer.from(traced.stdout)); proof.push('headers', Buffer.from(traced.stderr));
      assert.equal(proof.valid(count), true, 'decode/presentation order, zero reorder depth, no future references, closed IDRs');
      const keyframes = probe.frames.flatMap((frame: { key_frame: number }, index: number) => frame.key_frame ? [index] : []);
      assert.deepEqual(keyframes, [0, tier.frameRate * 4]);
      const decoded = await execute(ffmpeg, ['-v', 'error', '-i', file, '-f', 'null', '-']);
      assert.equal(decoded.stderr, '');
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('opt-in HKSV exposes only the cellular tier and retains the 15 fps choice', () => {
  const profile = streamProfile(1920, 1080);
  assert.deepEqual(recordingProfile(profile, false).resolutions, profile.resolutions);
  const shared = recordingProfile(profile, true);
  assert.deepEqual(shared.tiers, [selectCellularTier(profile.tiers)]);
  assert.deepEqual(shared.resolutions, [[1280, 720, 30], [1280, 720, 15]]);

  const low = recordingProfile(streamProfile(1080, 1080), true);
  assert.deepEqual(low.resolutions, [[720, 720, 30], [720, 720, 15]]);
});
