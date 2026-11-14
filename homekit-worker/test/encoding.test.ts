import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { encodeVideo, deviceArgs, selectEncoding, qualifyEncoding, EncodingUnavailableError } from '../src/encoding.js';
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
