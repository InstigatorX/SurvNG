import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, chmod, readFile, lstat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { connect } from 'node:net';
import { once } from 'node:events';
import { RtpHeader, RtpPacket } from 'werift';
import { FragmentRing, FragmentParser, producerReferenceTime, boxes } from '../src/fragments.js';
import { privateDirectory, identity, controlServer } from '../src/control.js';
import { SecureVideoSFrame, SFrameCipherSuite, SFrameReceiver, encodeHeader } from '../src/vendor/sframe.js';
import { HevcAccessUnitAssembler, SFrameRtpPacketizer } from '../src/vendor/sframeRtp.js';
import { videoOffer } from '../src/transports.js';
import { tiers } from '../src/media.js';

test('fragment retention has an explicit gap instead of skipping a slow reader', async () => {
  const ring = new FragmentRing(32, 6);
  ring.append(Buffer.alloc(8), 0); ring.append(Buffer.alloc(8), 2);
  const cursor = ring.cursor(); assert.equal(cursor, 1);
  ring.append(Buffer.alloc(8), 4); ring.append(Buffer.alloc(8), 6);
  assert.equal(ring.coverage, 6); assert.equal(ring.size, 24);
  await assert.rejects(ring.read(cursor, new AbortController().signal), /fell behind/);
  assert.throws(() => ring.append(Buffer.alloc(33), 8), /capacity/);
  assert.throws(() => ring.append(Buffer.alloc(1), 6), /discontinuity/);
});

test('fragment follow wakes on data, cancellation, and source closure', async () => {
  const ring = new FragmentRing();
  assert.throws(() => ring.cursor(), /not ready/);
  const abort = new AbortController();
  const next = ring.read(1, abort.signal); ring.append(Buffer.from('first'), 0);
  assert.equal((await next).data.toString(), 'first');
  const wait = ring.read(2, abort.signal); abort.abort(); await assert.rejects(wait);
  const closed = ring.read(2, new AbortController().signal); ring.close();
  await assert.rejects(closed, /closed/); assert.equal(ring.size, 0);
});

function box(name: string, payload = Buffer.alloc(0)): Buffer {
  const header = Buffer.alloc(8); header.writeUInt32BE(8 + payload.length); header.write(name, 4);
  return Buffer.concat([header, payload]);
}
test('MP4 parser accepts split reads and rejects corrupt or unbounded boxes', () => {
  const init: Buffer[] = []; const media: Buffer[] = [];
  const parser = new FragmentParser(value => init.push(value), value => media.push(value));
  const data = Buffer.concat([box('ftyp'), box('moov'), box('moof'), box('mdat', Buffer.from('payload'))]);
  for (const byte of data) parser.push(Buffer.from([byte]));
  assert.equal(init.length, 1); assert.equal(media.length, 1);
  assert.deepEqual(boxes(media[0]).map(value => value.type), ['moof', 'mdat']);
  const bad = Buffer.alloc(8); bad.writeUInt32BE(0xffffffff);
  assert.throws(() => parser.push(bad), /size/);
  assert.throws(() => boxes(Buffer.from('short')), /truncated/);
});

test('producer reference time preserves decode time and wall-clock milliseconds', () => {
  const data = producerReferenceTime(7, 123456789n, 1700000000125);
  assert.equal(data.toString('ascii', 4, 8), 'prft'); assert.equal(data[8], 1);
  assert.equal(data.readUInt32BE(12), 7); assert.equal(data.readBigUInt64BE(24), 123456789n);
  const ntp = data.readBigUInt64BE(16);
  assert.equal(ntp >> 32n, 1700000000n + 2208988800n); assert.equal(ntp & 0xffffffffn, 0x20000000n);
});

test('pairing identity is private and stable, and unsafe state is rejected', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-hksv-'));
  try {
    await privateDirectory(root); const first = await identity(root); const second = await identity(root);
    assert.deepEqual(first, second);
    assert.equal((await lstat(join(root, 'identity.json'))).mode & 0o777, 0o600);
    assert.equal(JSON.parse(await readFile(join(root, 'identity.json'), 'utf8')).id, first.id);
    await chmod(root, 0o755); await assert.rejects(privateDirectory(root), /0700/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('control endpoint validates framing and cannot replace a live endpoint', async () => {
  const root = await mkdtemp(join(tmpdir(), 'survng-hksv-'));
  const socket = join(root, 'test.sock');
  const server = await controlServer(socket, method => {
    if (method !== 'status') throw new Error('secret internal details'); return { ready: true };
  });
  const request = async (data: object) => {
    const client = connect(socket); client.on('error', () => {}); await once(client, 'connect');
    const chunks: Buffer[] = []; client.on('data', chunk => chunks.push(chunk));
    client.write(JSON.stringify(data) + '\n'); await once(client, 'end');
    return JSON.parse(Buffer.concat(chunks).toString());
  };
  try {
    assert.deepEqual(await request({ version: 1, id: 'a', method: 'status' }), { version: 1, id: 'a', result: { ready: true } });
    assert.equal((await request({ version: 2, id: 'a', method: 'status' })).error, 'invalid_request');
    assert.equal((await request({ version: 1, id: 'a', method: 'bad' })).error, 'request_failed');
    await assert.rejects(controlServer(socket, () => null), /EADDRINUSE/);
  } finally { await server.close(); await rm(root, { recursive: true, force: true }); }
});

test('SFrame authenticates payloads and isolates RTP streams', () => {
  const frame = new SecureVideoSFrame(true); const key = frame.senderKey!;
  const group = new Map([[key.kid, key.key]]);
  const sender = frame.videoStream(1234);
  const receiver = new SFrameReceiver(group, 1234, SFrameCipherSuite.AES_256_CTR_HMAC_SHA512_80);
  const clear = Buffer.from('encoded-video-frame'); const sealed = sender.protectFrame(clear);
  assert.deepEqual(receiver.unprotectFrame(sealed), clear);
  assert.notDeepEqual(sender.protectFrame(clear), sealed);
  const corrupt = Buffer.from(sealed); corrupt[corrupt.length - 1] ^= 1;
  assert.throws(() => receiver.unprotectFrame(corrupt), /auth/);
  assert.throws(() => receiver.unprotectFrame(Buffer.from([0xff])), /truncated/);
  assert.throws(() => new SFrameReceiver(group, 5678, SFrameCipherSuite.AES_256_CTR_HMAC_SHA512_80).unprotectFrame(sealed), /auth/);
  assert.throws(() => encodeHeader(-1n, 0n), /range/);
  assert.throws(() => encodeHeader(0n, 1n << 64n), /range/);
});

test('HEVC access units retain NAL lengths and SFrame packets remain bounded', () => {
  const assembler = new HevcAccessUnitAssembler();
  const nal = Buffer.from([0x26, 0x01, 3, 4, 5]);
  const packet = new RtpPacket(new RtpHeader({ ssrc: 1, timestamp: 90, marker: true }), nal);
  const frame = assembler.push(packet)!;
  assert.equal(frame.data.readUInt32BE(0), nal.length); assert.deepEqual(frame.data.subarray(4), nal);
  const packetizer = new SFrameRtpPacketizer({ ssrc: 42, payloadType: 99, maxPayload: 1200 });
  const packets = packetizer.packetize(Buffer.alloc(3000), 90, true);
  assert.ok(packets.length > 1); assert.ok(packets.every(value => value.payload.length <= 1200));
  assert.equal(packets.at(-1)!.header.marker, true);
});

test('remote SDP has tier bounds and a unique RTP stream extension', () => {
  const offer = videoOffer('v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF 99\r\nc=IN IP4 0.0.0.0\r\na=extmap:1 existing\r\nm=audio 9 UDP/TLS/RTP/SAVPF 110\r\n', tiers[0]);
  assert.match(offer, /b=TIAS:1800000/); assert.match(offer, /a=extmap:2 urn:ietf/);
  assert.match(offer, /max-width=1920;max-height=1080/);
  assert.ok(offer.indexOf('a=simulcast:send 1') < offer.indexOf('m=audio'));
});
