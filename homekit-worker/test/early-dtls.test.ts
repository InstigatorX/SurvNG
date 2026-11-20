import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Event, MediaStreamTrack, RTCDtlsTransport, RTCPeerConnection, type CandidatePair, type Protocol } from 'werift';
import { preserveEarlyClientHello } from '../src/early-dtls.js';

function hello(mark: number) {
  const body = Buffer.concat([
    Buffer.from([0xfe, 0xfd]), Buffer.alloc(32, mark),
    // Empty session and cookie, ECDHE/ECDSA AES-128-GCM, null compression,
    // P-256 supported groups and ECDSA/RSA SHA-256 signature algorithms.
    Buffer.from('00000002c02b01000012000a000400020017000d0006000404010403', 'hex'),
  ]);
  const header = Buffer.alloc(25);
  header.set([22, 0xfe, 0xfd]);
  header.writeUInt16BE(body.length + 12, 11);
  header[13] = 1;
  header.writeUIntBE(body.length, 14, 3);
  header.writeUIntBE(body.length, 22, 3);
  return Buffer.concat([header, body]);
}

type Address = readonly [string, number];

function fixture() {
  const address = ['192.0.2.2', 4000] as const;
  const selected = { onDataReceived: new Event<[Buffer, Address]>() };
  const other = { onDataReceived: new Event<[Buffer, Address]>() };
  const pair = { protocol: selected, remoteAddr: address };
  const connection = { nominated: pair };
  const received: Buffer[] = [];
  const transport = {
    iceTransport: { connection },
    state: 'new',
    role: 'auto',
    onStateChange: new Event<[string]>(),
    dtls: undefined as object | undefined,
  };
  let replays = 0;
  const buffer = preserveEarlyClientHello(transport as unknown as RTCDtlsTransport, () => replays++);
  const observe = (protocol: typeof selected) => buffer.observe({ protocol: protocol as unknown as Protocol } as CandidatePair);
  observe(selected);
  observe(selected);
  observe(other);
  const start = (role = 'server') => {
    transport.state = 'connecting';
    transport.role = role;
    transport.onStateChange.execute('connecting');
    transport.dtls = { transport: { socket: { onData: (data: Buffer) => received.push(data) } } };
  };
  return { address, selected, other, connection, received, buffer, start, replays: () => replays,
    stateChanges: transport.onStateChange };
}

test('replays one early ClientHello from the nominated ICE address and protocol', async () => {
  const f = fixture();
  assert.equal(f.selected.onDataReceived.length, 1);
  f.other.onDataReceived.execute(hello(1), f.address);
  f.selected.onDataReceived.execute(hello(2), [f.address[0], 4001]);
  f.selected.onDataReceived.execute(hello(3).subarray(0, 65), f.address);
  f.selected.onDataReceived.execute(hello(4), f.address);
  f.selected.onDataReceived.execute(hello(5), f.address);
  assert.deepEqual(f.received, []);
  f.start();
  await Promise.resolve();

  assert.deepEqual(f.received, [hello(4)]);
  assert.equal(f.replays(), 1);
  assert.equal(f.selected.onDataReceived.length + f.other.onDataReceived.length + f.stateChanges.length, 0);
});

test('discards early ClientHello on peer change, client role, or shutdown', async () => {
  for (const reason of ['peer changed', 'client role', 'stopped'] as const) {
    const f = fixture();
    f.selected.onDataReceived.execute(hello(1), f.address);
    f.start(reason === 'client role' ? 'client' : 'server');
    if (reason === 'peer changed') f.connection.nominated = { protocol: f.other, remoteAddr: f.address };
    if (reason === 'stopped') f.buffer.stop();
    await Promise.resolve();

    assert.deepEqual(f.received, [], reason);
    assert.equal(f.replays(), 0, reason);
  }
});

test('werift sends a HelloVerifyRequest for an early packet without a peer retry', { timeout: 5000 }, async () => {
  const pc = new RTCPeerConnection();
  const dtls = pc.addTransceiver(new MediaStreamTrack({ kind: 'video' }), { direction: 'sendonly' }).dtlsTransport;
  const ice = dtls.iceTransport.connection;
  const protocol = { onDataReceived: new Event<[Buffer, Address]>() };
  const address = ['127.0.0.1', 12345] as const;
  const pair = { protocol, remoteAddr: address } as unknown as CandidatePair;
  const replies: Buffer[] = [];
  ice.send = async packet => { replies.push(packet); };
  dtls.role = 'server';
  dtls.localCertificate = await RTCDtlsTransport.SetupCertificate();
  // This probe stops at the server's cookie challenge, before peer authentication.
  dtls.setRemoteParams(dtls.localParameters);
  const buffer = preserveEarlyClientHello(dtls, () => {});
  ice.nominated = pair;
  buffer.observe(pair);
  try {
    const packet = hello(1);
    protocol.onDataReceived.execute(packet, address);
    ice.onData.execute(packet);
    assert.equal(replies.length, 0);
    const starting = dtls.start();
    void starting.catch(() => {});
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.equal(replies.length, 1);
    assert.equal(replies[0][13], 3); // HelloVerifyRequest
  } finally {
    buffer.stop();
    ice.nominated = undefined;
    await pc.close();
  }
});
