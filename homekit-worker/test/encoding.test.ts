import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { encodeVideo, deviceArgs, selectEncoding, qualifyEncoding, EncodingUnavailableError } from '../src/encoding.js';
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

test('QSV produces decodable Main HEVC with periodic IDRs and no B pictures at every tier', {
  skip: process.env.HKSV_QSV_TESTS !== '1', timeout: 60000,
}, async () => {
  const execute = promisify(execFile);
  const ffmpeg = process.env.HKSV_FFMPEG || 'ffmpeg';
  const encoding = { backend: 'qsv' as const, device: '/dev/dri/renderD128' };
  const root = await mkdtemp(join(tmpdir(), 'survng-qsv-'));
  try {
    assert.equal(await qualifyEncoding(ffmpeg, root, async () => {}, new AbortController().signal, encoding), true);
    for (const tier of tiers) {
      const file = join(root, `${tier.identifier}.hevc`);
      const count = tier.frameRate * 4 + 5;
      await execute(ffmpeg, ['-hide_banner', '-loglevel', 'error', ...deviceArgs(encoding),
        '-f', 'lavfi', '-i', `testsrc2=size=${tier.width}x${tier.height}:rate=${tier.frameRate}`, '-frames:v', String(count),
        ...encodeVideo(tier, encoding), '-f', 'hevc', file], { timeout: 20000 });
      const probe = JSON.parse((await execute('ffprobe', ['-v', 'error', '-show_streams', '-show_frames', '-of', 'json', file], { maxBuffer: 4 * 1024 * 1024 })).stdout);
      assert.equal(probe.streams[0].codec_name, 'hevc'); assert.equal(probe.streams[0].profile, 'Main');
      assert.equal(probe.streams[0].width, tier.width); assert.equal(probe.streams[0].height, tier.height);
      assert.equal(probe.frames.length, count);
      assert.equal(probe.frames.filter((frame: { pict_type: string }) => frame.pict_type === 'B').length, 0);
      const keyframes = probe.frames.flatMap((frame: { key_frame: number }, index: number) => frame.key_frame ? [index] : []);
      assert.deepEqual(keyframes, [0, tier.frameRate * 4]);
      const data = await readFile(file);
      const types = new Map<number, number>();
      for (let offset = 0; offset + 5 < data.length;) {
        const start = data.indexOf(Buffer.from([0, 0, 1]), offset);
        if (start < 0) break;
        const type = (data[start + 3] >> 1) & 0x3f;
        types.set(type, (types.get(type) || 0) + 1); offset = start + 3;
      }
      assert.ok((types.get(19) || 0) + (types.get(20) || 0) >= 2, 'every GOP starts with an IDR');
      assert.equal(types.has(21), false, 'no open-GOP CRA pictures');
      const decoded = await execute(ffmpeg, ['-v', 'error', '-i', file, '-f', 'null', '-']);
      assert.equal(decoded.stderr, '');
    }
  } finally { await rm(root, { recursive: true, force: true }); }
});
