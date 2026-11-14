import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawn } from 'node:child_process';
import { createConnection } from 'node:net';
import { CameraSource, TsReplay } from '../src/camera-source.js';

function packet(pid: number, payload: Buffer, pusi = false): Buffer {
  const buf = Buffer.alloc(188, 0xff);
  buf[0] = 0x47;
  buf[1] = (pusi ? 0x40 : 0) | ((pid >> 8) & 0x1f);
  buf[2] = pid & 0xff;
  buf[3] = 0x10;
  buf.set(payload.subarray(0, 184), 4);
  return buf;
}

function pat(): Buffer {
  return packet(0, Buffer.from([
    0x00, 0x00, 0xb0, 0x0d, 0x00, 0x01, 0xc1, 0x00, 0x00,
    0x00, 0x01, 0xe0, 0x10, 0x00, 0x00, 0x00, 0x00,
  ]), true);
}

function pmt(): Buffer {
  return packet(0x10, Buffer.from([
    0x00, 0x02, 0xb0, 0x12, 0x00, 0x01, 0xc1, 0x00, 0x00,
    0xe1, 0x00, 0xf0, 0x00, 0x24, 0xe1, 0x00, 0xf0, 0x00,
    0x00, 0x00, 0x00, 0x00,
  ]), true);
}

function video(nal: number, marker: number): Buffer {
  const pes = Buffer.alloc(40, marker);
  pes.set([0x00, 0x00, 0x01, 0xe0, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, nal << 1], 0);
  return packet(0x100, pes, true);
}

test('the replay handed to a new viewer starts at the latest keyframe', () => {
  const replay = new TsReplay();
  replay.push(Buffer.concat([pat(), pmt(), video(1, 0x11)]));
  assert.equal(replay.snapshot().length, 0);
  replay.push(video(32, 0x22));
  replay.push(video(1, 0x33));
  const first = replay.snapshot();
  assert.equal(first[0], 0x47);
  assert.equal(((first[1] & 0x1f) << 8) | first[2], 0);
  assert.equal(first.includes(Buffer.from([0x22])), true);
  assert.equal(first.includes(Buffer.from([0x11])), false);
  replay.push(video(19, 0x44));
  const next = replay.snapshot();
  assert.equal(next.includes(Buffer.from([0x44])), true);
  assert.equal(next.includes(Buffer.from([0x22])), false);
  assert.equal(next.includes(Buffer.from([0x33])), false);
});

test('a viewer that joins mid-GOP is given the keyframe already produced', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-source-'));
  const path = join(root, 'source.sock');
  const before = Buffer.concat([pat(), pmt(), video(1, 0x11)]);
  const key = video(32, 0x22);
  const source = new CameraSource(path, async () => spawn(process.execPath, ['-e', `
    const before = Buffer.from(${JSON.stringify(before.toString('base64'))}, 'base64');
    const key = Buffer.from(${JSON.stringify(key.toString('base64'))}, 'base64');
    process.stdout.write(before);
    setTimeout(() => process.stdout.write(key), 30);
    setInterval(() => {}, 1000);
  `]));
  try {
    await source.start();
    const first = createConnection(path);
    const seen: Buffer[] = [];
    first.on('data', chunk => seen.push(chunk));
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('keyframe was not produced')), 2000);
      first.on('data', () => {
        if (Buffer.concat(seen).includes(0x22)) { clearTimeout(timer); resolve(); }
      });
    });
    const late = createConnection(path);
    const lateData: Buffer[] = [];
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('late viewer missed the keyframe')), 2000);
      late.on('data', chunk => {
        lateData.push(chunk);
        const joined = Buffer.concat(lateData);
        if (!joined.includes(0x22)) return;
        clearTimeout(timer);
        assert.equal(((joined[1] & 0x1f) << 8) | joined[2], 0, 'join starts with the program table');
        assert.equal(joined.includes(0x11), false);
        resolve();
      });
    });
    first.destroy(); late.destroy();
  } finally {
    await source.stop();
    await rm(root, { recursive: true, force: true });
  }
});
