import type { CandidatePair, Protocol, RTCDtlsTransport } from 'werift';

const MAX_PACKET_AGE_MS = 1000;

type PeerAddress = readonly [string, number];

function clientHello(data: Buffer): Buffer | undefined {
  // A DTLS 1.x handshake record with a complete, unfragmented ClientHello.
  if (data.length < 25 || data.length > 1500 || data[0] !== 22 || data[1] !== 0xfe
      || ![0xff, 0xfd].includes(data[2]) || data.readUInt16BE(3) !== 0 || data[13] !== 1) return;
  const recordSize = data.readUInt16BE(11);
  const messageSize = data.readUIntBE(14, 3);
  if (messageSize < 42 || recordSize !== messageSize + 12 || 13 + recordSize > data.length
      || data.readUIntBE(19, 3) !== 0 || data.readUIntBE(22, 3) !== messageSize) return;
  const end = 13 + recordSize;
  if (data[25] !== 0xfe || ![0xff, 0xfd].includes(data[26])) return;
  // Skip the version and random, then check each length-prefixed hello vector.
  let offset = 59;
  const sessionIdSize = data[offset++];
  if (sessionIdSize > 32 || offset + sessionIdSize >= end) return;
  offset += sessionIdSize;
  offset += 1 + data[offset]; // Cookie
  if (offset + 2 > end) return;
  const cipherSize = data.readUInt16BE(offset);
  if (cipherSize < 2 || cipherSize % 2 || offset + 2 + cipherSize >= end) return;
  offset += 2 + cipherSize;
  const compressionSize = data[offset++];
  if (!compressionSize || offset + compressionSize > end) return;
  offset += compressionSize;
  if (offset !== end && (offset + 2 > end || offset + 2 + data.readUInt16BE(offset) !== end)) return;
  return Buffer.from(data.subarray(0, end));
}

export function preserveEarlyClientHello(transport: RTCDtlsTransport, onReplay: () => void) {
  const connection = transport.iceTransport.connection;
  const subscriptions = new Map<Protocol, () => void>();
  let pending: { packet: Buffer; protocol: Protocol; address: PeerAddress; capturedAt: number } | undefined;
  let stopped = false;
  let listening = true;

  const matches = (protocol: Protocol, address: PeerAddress) => {
    const pair = connection.nominated;
    return pair?.protocol === protocol && pair.remoteAddr[0] === address[0] && pair.remoteAddr[1] === address[1];
  };
  const detach = () => {
    listening = false;
    stateChange.unSubscribe();
    for (const unsubscribe of subscriptions.values()) unsubscribe();
    subscriptions.clear();
  };
  const stateChange = transport.onStateChange.subscribe(state => {
    if (state === 'new') return;
    detach();
    const hello = pending;
    pending = undefined;
    if (state !== 'connecting' || transport.role !== 'server' || !hello) return;
    // Werift announces connecting just before constructing its DTLS adapter.
    queueMicrotask(() => {
      const socket = transport.dtls?.transport.socket;
      if (stopped || !socket || transport.state !== 'connecting' || performance.now() - hello.capturedAt > MAX_PACKET_AGE_MS
          || !matches(hello.protocol, hello.address)) return;
      socket.onData(hello.packet, hello.address);
      onReplay();
    });
  });

  return {
    observe(pair: CandidatePair) {
      const { protocol } = pair;
      if (!listening || subscriptions.has(protocol) || subscriptions.size >= 16) return;
      const subscription = protocol.onDataReceived.subscribe((data, address) => {
        if (!address || transport.state !== 'new' || !matches(protocol, address)) return;
        const now = performance.now();
        if (pending && now - pending.capturedAt <= MAX_PACKET_AGE_MS && matches(pending.protocol, pending.address)) return;
        const packet = clientHello(data);
        if (packet) pending = { packet, protocol, address: [address[0], address[1]], capturedAt: now };
      });
      subscriptions.set(protocol, subscription.unSubscribe);
    },
    stop() {
      stopped = true;
      detach();
      pending = undefined;
    },
  };
}
