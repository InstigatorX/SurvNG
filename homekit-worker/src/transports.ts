import { randomBytes, randomInt } from 'node:crypto';
import { isIP } from 'node:net';
import type { Socket } from 'node:dgram';
import { MediaStreamTrack, RTCPeerConnection, RTCRtpCodecParameters, RTCRtpHeaderExtensionParameters, RtpPacket, SrtpSession, SrtcpSession, RtcpSrPacket, RtcpSenderInfo } from 'werift';
import type { MultiTierRTPStreamingDelegate, MultiTierPrepareStreamRequest, MultiTierPrepareStreamResponse,
  MultiTierStreamStartRequest, WebRTCStreamingDelegate, WebRTCSolicitOfferRequest, WebRTCOffer,
  WebRTCProvideAnswerRequest, WebRTCReofferRequest, WebRTCReofferAnswer, WebRTCUpdateSessionRequest } from '@homebridge/hap-nodejs';
import { SyntheticMedia, RtpFeed, tiers, udp, type Tier } from './media.js';
import { SecureVideoSFrame } from './vendor/sframe.js';
import { HevcAccessUnitAssembler, SFrameRtpPacketizer } from './vendor/sframeRtp.js';

type Crypto = { cryptoSuite: number; masterKey: Buffer; masterSalt: Buffer };
export const STREAM_ID_URI = 'urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id';
function crypto(): Crypto { return { cryptoSuite: 0, masterKey: randomBytes(16), masterSalt: randomBytes(14) }; }
function keys(value: Crypto) {
  return { profile: 1, keys: { localMasterKey: value.masterKey, localMasterSalt: value.masterSalt,
    remoteMasterKey: value.masterKey, remoteMasterSalt: value.masterSalt } };
}
interface LocalSession {
  sockets: Socket[]; response: MultiTierPrepareStreamResponse; request: MultiTierPrepareStreamRequest;
  feed?: RtpFeed; timeout: NodeJS.Timeout; reports?: NodeJS.Timeout; closed: boolean;
  stopping?: Promise<void>;
}

export class LocalStreams implements MultiTierRTPStreamingDelegate {
  readonly sessions = new Map<string, LocalSession>();
  private preparing = new Set<string>();
  private generation = 0;
  readonly counters = { started: 0, failures: 0 };
  constructor(private media: SyntheticMedia, private allowed: () => boolean) {}
  async prepareStream(request: MultiTierPrepareStreamRequest): Promise<MultiTierPrepareStreamResponse> {
    const id = request.sessionIdentifier;
    if (!this.allowed() || this.sessions.size + this.preparing.size >= 5 || this.sessions.has(id) || this.preparing.has(id)) throw new Error('local session unavailable');
    if (!isIP(request.targetAddress) || !isIP(request.sourceAddress)) throw new Error('invalid stream address');
    for (const value of [request.video, request.audio]) {
      if (value.cryptoSuite !== 0 || value.masterKey.length !== 16 || value.masterSalt.length !== 14) throw new Error('unsupported SRTP parameters');
    }
    for (const port of [request.controllerVideoPort, request.controllerAudioPort]) {
      if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('invalid stream port');
    }
    this.preparing.add(id);
    const generation = this.generation;
    const sockets: Socket[] = [];
    try {
      sockets.push(await udp(request.sourceAddress)); sockets.push(await udp(request.sourceAddress));
      if (!this.allowed() || generation !== this.generation) throw new Error('streaming disabled');
      const response = { addressOverride: request.sourceAddress,
        videoPort: sockets[0].address().port, audioPort: sockets[1].address().port,
        videoSSRC: randomInt(1, 0x7fffffff), audioSSRC: randomInt(1, 0x7fffffff), video: crypto(), audio: crypto() };
      const session: LocalSession = { request, response, sockets, closed: false,
        timeout: setTimeout(() => void this.stopStream(id), 60_000) };
      this.sessions.set(id, session);
      sockets.forEach(socket => socket.on('error', () => {
        if (!session.closed) this.counters.failures++;
        void this.stopStream(id);
      }));
      return response;
    } catch (error) { sockets.forEach(socket => socket.close()); throw error; }
    finally { this.preparing.delete(id); }
  }
  async startStream(request: MultiTierStreamStartRequest): Promise<void> {
    const id = request.sessionIdentifier;
    const session = this.sessions.get(id);
    const tier = tiers.find(t => t.identifier === request.videoTier);
    if (!session || session.feed || !tier || !this.allowed()) throw new Error('local stream unavailable');
    clearTimeout(session.timeout);
    const fail = () => { if (!session.closed) this.counters.failures++; void this.stopStream(id); };
    const response = session.response;
    const states = [response.video, response.audio].map((value, index) => ({
      srtp: new SrtpSession(keys(value)), srtcp: new SrtcpSession(keys(value)),
      ssrc: index === 0 ? response.videoSSRC : response.audioSSRC,
      count: 0, bytes: 0, timestamp: 0, wall: 0,
    }));
    // The controller receives the SSRC returned by prepare, not the later START hint.
    const send = (index: number, packet: RtpPacket) => {
      if (session.closed) return;
      const state = states[index];
      packet.header.ssrc = state.ssrc; packet.header.payloadType = index === 0 ? 99 : 110;
      state.count++; state.bytes += packet.payload.length; state.timestamp = packet.header.timestamp; state.wall = Date.now();
      session.sockets[index].send(state.srtp.encrypt(packet.payload, packet.header),
        index === 0 ? session.request.controllerVideoPort : session.request.controllerAudioPort,
        session.request.targetAddress, error => { if (error) fail(); });
    };
    session.reports = setInterval(() => {
      states.forEach((state, index) => {
        if (!state.count || session.closed) return;
        const ms = BigInt(state.wall);
        const sr = new RtcpSrPacket({ ssrc: state.ssrc, senderInfo: new RtcpSenderInfo({
          ntpTimestamp: ((ms / 1000n + 2208988800n) << 32n) | ((ms % 1000n) << 32n) / 1000n,
          rtpTimestamp: state.timestamp, packetCount: state.count >>> 0, octetCount: state.bytes >>> 0,
        }) });
        session.sockets[index].send(state.srtcp.encrypt(sr.serialize()),
          index === 0 ? session.request.controllerVideoPort : session.request.controllerAudioPort,
          session.request.targetAddress, error => { if (error) fail(); });
      });
    }, 500);
    // Bound abandoned lab sessions even if the controller never sends END.
    session.timeout = setTimeout(fail, 10 * 60_000);
    session.feed = new RtpFeed(this.media, tier);
    try { await session.feed.start(packet => send(0, packet), packet => send(1, packet), fail); this.counters.started++; }
    catch (error) { await this.stopStream(id); throw error; }
  }
  async stopStream(id: string): Promise<void> {
    const session = this.sessions.get(id);
    if (!session) return;
    if (session.stopping) return session.stopping;
    session.closed = true; clearTimeout(session.timeout); clearInterval(session.reports);
    session.stopping = (async () => {
      try { await session.feed?.stop(); }
      finally { session.sockets.forEach(socket => socket.close()); this.sessions.delete(id); }
    })();
    return session.stopping;
  }
  async close(): Promise<void> { this.generation++; await Promise.all([...this.sessions.keys()].map(id => this.stopStream(id))); }
}

interface RemoteSession {
  pc: RTCPeerConnection; frame: SecureVideoSFrame; feed?: RtpFeed; closed: boolean;
  timeout: NodeJS.Timeout; answered: boolean;
  stopping?: Promise<void>;
}

/**
 * Apple's video offer needs bitrate and an RTP stream ID. Adapted from camera.ui
 * webrtcSessions.ts at 91ce370391c0f18e63faeb26a876967315a317bc; MIT copyright
 * seydx. See vendor/LICENSE.camera-ui.md. SurvNG also sends the negotiated RID
 * on RTP packets and uses a single bundled transport with owned teardown.
 */
export function videoOffer(sdp: string, tier: Tier): string {
  const lines = sdp.split(/\r?\n/);
  const start = lines.findIndex(line => line.startsWith('m=video'));
  if (start < 0) throw new Error('missing video SDP');
  let end = lines.findIndex((line, i) => i > start && line.startsWith('m='));
  if (end < 0) end = lines.length;
  const connection = lines.findIndex((line, i) => i > start && i < end && line.startsWith('c='));
  lines.splice(connection + 1, 0, `b=AS:${tier.peakBitrate}`, `b=TIAS:${tier.peakBitrate * 1000}`); end += 2;
  const ids = new Set(lines.flatMap(line => { const match = /^a=extmap:(\d+)/.exec(line); return match ? [Number(match[1])] : []; }));
  let id = 1; while (ids.has(id)) id++;
  if (id > 14) throw new Error('no RTP extension id available');
  const existing = lines.some(line => line.startsWith('a=extmap:') && line.endsWith(` ${STREAM_ID_URI}`));
  lines.splice(end, 0, ...(existing ? [] : [`a=extmap:${id} ${STREAM_ID_URI}`]),
    `a=rid:1 send max-width=${tier.width};max-height=${tier.height};max-fps=${tier.frameRate};max-br=${tier.peakBitrate * 1000}`,
    'a=simulcast:send 1');
  return lines.join('\r\n');
}

export class RemoteStreams implements WebRTCStreamingDelegate {
  readonly sessions = new Map<string, RemoteSession>();
  readonly counters = { started: 0, failures: 0 };
  constructor(private media: SyntheticMedia, private allowed: () => boolean, private address: string) {}
  async handleSolicitOffer(request: WebRTCSolicitOfferRequest): Promise<WebRTCOffer> {
    const id = request.sessionIdentifier;
    if (!this.allowed() || this.sessions.size >= 6 || this.sessions.has(id)) throw new Error('remote session unavailable');
    // Apple uses BUNDLE. max-compat gathers a second transport which this werift
    // revision loses when applying a bundled answer, leaving its UDP sockets open.
    const pc = new RTCPeerConnection({ bundlePolicy: 'max-bundle', iceServers: [], iceUseIpv6: false,
      iceAdditionalHostAddresses: [this.address], iceInterfaceAddresses: { udp4: this.address },
      headerExtensions: { video: [new RTCRtpHeaderExtensionParameters({ id: 1, uri: STREAM_ID_URI })], audio: [] }, codecs: {
      video: [new RTCRtpCodecParameters({ mimeType: 'video/H265', clockRate: 90000, payloadType: 99,
        parameters: 'profile-id=1;tier-flag=0;level-id=153;tx-mode=SRST',
        rtcpFeedback: [{ type: 'nack' }, { type: 'nack', parameter: 'pli' }, { type: 'ccm', parameter: 'fir' }] })],
      audio: [new RTCRtpCodecParameters({ mimeType: 'audio/opus', clockRate: 48000, channels: 2, payloadType: 110,
        parameters: 'minptime=20;useinbandfec=1;stereo=0;sprop-stereo=0' })],
    } });
    const vtrack = new MediaStreamTrack({ kind: 'video' }); const atrack = new MediaStreamTrack({ kind: 'audio' });
    const video = pc.addTransceiver(vtrack, { direction: 'sendonly' });
    const audio = pc.addTransceiver(atrack, { direction: 'sendonly' });
    const fail = () => { const current = this.sessions.get(id); if (current && !current.closed) this.counters.failures++; void this.handleEndSession(id); };
    const session: RemoteSession = { pc, frame: new SecureVideoSFrame(true), closed: false, answered: false,
      timeout: setTimeout(fail, 60_000) };
    this.sessions.set(id, session);
    const candidates: NonNullable<WebRTCOffer['candidates']> = [];
    pc.onIceCandidate.subscribe(candidate => {
      if (!candidate) return;
      const value = candidate.toJSON();
      candidates.push({ candidate: value.candidate, sdpMid: value.sdpMid ?? undefined, sdpMLineIndex: value.sdpMLineIndex ?? undefined });
    });
    pc.connectionStateChange.subscribe(state => {
      if (state === 'failed' || state === 'disconnected' || state === 'closed') fail();
      if (state !== 'connected' || session.closed || session.feed) return;
      if (!this.allowed()) { fail(); return; }
      clearTimeout(session.timeout); session.timeout = setTimeout(fail, 10 * 60_000);
      const assembler = new HevcAccessUnitAssembler();
      const vc = session.frame.videoStream(video.sender.ssrc); const ac = session.frame.audioStream(audio.sender.ssrc);
      const vp = new SFrameRtpPacketizer({ ssrc: video.sender.ssrc, payloadType: video.sender.codec?.payloadType ?? 99, maxPayload: 1200 });
      const ap = new SFrameRtpPacketizer({ ssrc: audio.sender.ssrc, payloadType: audio.sender.codec?.payloadType ?? 110, maxPayload: 1200 });
      session.feed = new RtpFeed(this.media, tiers[0]);
      this.counters.started++;
      void session.feed.start(packet => {
        const frame = assembler.push(packet);
        if (frame) for (const value of vp.packetize(vc.protectFrame(frame.data), frame.timestamp, true)) {
          const extension = video.headerExtensions.find(extension => extension.uri === STREAM_ID_URI);
          if (extension) value.header.extensions.push({ id: extension.id, payload: Buffer.from('1') });
          vtrack.writeRtp(value);
        }
      }, packet => {
        for (const value of ap.packetize(ac.protectFrame(packet.payload), packet.header.timestamp, packet.header.marker)) atrack.writeRtp(value);
      }, fail).catch(fail);
    });
    try {
      const offer = await pc.createOffer(); await pc.setLocalDescription(offer);
      if (session.closed || !this.allowed()) throw new Error('remote session cancelled');
      return { sdpOffer: videoOffer(pc.localDescription!.sdp, tiers[0]), candidates, sframe: session.frame.senderKey };
    } catch (error) { await this.handleEndSession(id); throw error; }
  }
  async handleProvideAnswer(request: WebRTCProvideAnswerRequest): Promise<void> {
    const session = this.sessions.get(request.sessionIdentifier);
    if (!session || session.answered || session.closed) throw new Error('unknown remote session');
    try {
      await session.pc.setRemoteDescription({ type: 'answer', sdp: request.sdpAnswer });
      session.answered = true;
      for (const candidate of request.candidates) await session.pc.addIceCandidate(candidate);
    } catch (error) { await this.handleEndSession(request.sessionIdentifier); throw error; }
  }
  async handleReoffer(request: WebRTCReofferRequest): Promise<WebRTCReofferAnswer> {
    const session = this.sessions.get(request.sessionIdentifier);
    if (!session || session.closed) throw new Error('unknown remote session');
    try {
      await session.pc.setRemoteDescription({ type: 'offer', sdp: request.sdpOffer });
      await session.pc.setLocalDescription(await session.pc.createAnswer());
      return { sdpAnswer: session.pc.localDescription!.sdp };
    } catch (error) { await this.handleEndSession(request.sessionIdentifier); throw error; }
  }
  async handleUpdateSession(request: WebRTCUpdateSessionRequest): Promise<void> {
    const session = this.sessions.get(request.sessionIdentifier);
    if (!session || session.closed) throw new Error('unknown remote session');
    // No talkback is advertised, but handle the controller's receive-key lifecycle.
    session.frame.addReceiveKeys(request.receiveKeysToAdd); session.frame.removeReceiveKeys(request.receiveKIDsToRemove);
  }
  async handleEndSession(id: string): Promise<void> {
    const session = this.sessions.get(id);
    if (!session) return;
    if (session.stopping) return session.stopping;
    if (session.closed) return; // close() can emit synchronously before the promise is assigned.
    session.closed = true; clearTimeout(session.timeout);
    session.stopping = (async () => {
      try { await session.feed?.stop(); }
      finally { await session.pc.close(); this.sessions.delete(id); }
    })();
    return session.stopping;
  }
  async close(): Promise<void> { await Promise.all([...this.sessions.keys()].map(id => this.handleEndSession(id))); }
}
