import { randomInt } from 'node:crypto';
import { isIP } from 'node:net';
import type { Socket } from 'node:dgram';
import { MediaStreamTrack, RTCPeerConnection, RTCRtpCodecParameters, RTCRtpHeaderExtensionParameters, RtpPacket, SrtpSession, SrtcpSession, RtcpSrPacket, RtcpSenderInfo, RtcpPacketConverter, isRtcp } from 'werift';
import type { MultiTierRTPStreamingDelegate, MultiTierPrepareStreamRequest, MultiTierPrepareStreamResponse,
  MultiTierStreamStartRequest, WebRTCStreamingDelegate, WebRTCSolicitOfferRequest, WebRTCOffer,
  WebRTCProvideAnswerRequest, WebRTCReofferRequest, WebRTCReofferAnswer, WebRTCUpdateSessionRequest } from '@homebridge/hap-nodejs';
import { type MediaSource, RtpFeed, tiers, udp, type Tier } from './media.js';
import { SecureVideoSFrame } from './vendor/sframe.js';
import { HevcAccessUnitAssembler, SFrameRtpPacketizer } from './vendor/sframeRtp.js';
import { ReceiverFeedback } from './diagnostics.js';
import { preserveEarlyClientHello } from './early-dtls.js';

type Crypto = { cryptoSuite: number; masterKey: Buffer; masterSalt: Buffer };
export const STREAM_ID_URI = 'urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id';
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
  readonly feedback = { video: new ReceiverFeedback(90000), audio: new ReceiverFeedback(48000), invalidPackets: 0 };
  readonly sent = { videoPackets: 0, audioPackets: 0 };
  negotiation?: { videoTier: number; audioTier?: number; videoSSRCMatchesPrepared: boolean; audioSSRCMatchesPrepared: boolean };
  constructor(private media: MediaSource, private allowed: () => boolean, private catalog: readonly Tier[] = tiers) {}
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
        // HKSV3 uses the controller's negotiated SRTP parameters in both directions.
        videoSSRC: randomInt(1, 0x7fffffff), audioSSRC: randomInt(1, 0x7fffffff), video: request.video, audio: request.audio };
      const session: LocalSession = { request, response, sockets, closed: false,
        timeout: setTimeout(() => void this.stopStream(id), 60_000) };
      const incoming = [request.video, request.audio].map(value => new SrtcpSession(keys(value)));
      sockets.forEach((socket, index) => socket.on('message', data => {
        if (session.closed || !isRtcp(data)) return;
        try {
          for (const packet of RtcpPacketConverter.deSerialize(incoming[index].decrypt(data))) {
            // A compound report may cover both media streams on either socket.
            this.feedback.video.observe(packet, response.videoSSRC);
            this.feedback.audio.observe(packet, response.audioSSRC);
          }
        } catch { this.feedback.invalidPackets++; }
      }));
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
    const tier = this.catalog.find(t => t.identifier === request.videoTier);
    if (!session || session.feed || !tier || !this.allowed()) throw new Error('local stream unavailable');
    this.negotiation = { videoTier: request.videoTier, audioTier: request.audioTier,
      videoSSRCMatchesPrepared: request.videoSSRC === session.response.videoSSRC,
      audioSSRCMatchesPrepared: request.audioSSRC === session.response.audioSSRC };
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
      if (index === 0) this.sent.videoPackets++; else this.sent.audioPackets++;
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
  timeout: NodeJS.Timeout; offerReady: boolean; signaling?: Promise<unknown>;
  stopping?: Promise<void>;
  earlyDtls: ReturnType<typeof preserveEarlyClientHello>;
}

/**
 * Apple's video offer needs bitrate and an RTP stream ID. Adapted from camera.ui
 * webrtcSessions.ts at 91ce370391c0f18e63faeb26a876967315a317bc; MIT copyright
 * seydx. See vendor/LICENSE.camera-ui.md. SurvNG also sends the negotiated RID
 * in signaling and uses a single bundled transport with owned teardown.
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
  readonly counters = { solicitations: 0, offers: 0, answers: 0, reoffers: 0, started: 0,
    failures: 0, videoPackets: 0, audioPackets: 0, lastState: 'idle',
    lastIceState: 'idle', lastDtlsState: 'idle', lastFailure: '', mediaReady: 0,
    lastOfferTimeMs: 0, lastConnectTimeMs: 0, lastLocalCandidates: 0, lastRemoteCandidates: 0,
    lastVideoTier: null as { width: number; height: number; peakBitrate: number } | null,
    earlyClientHellosReplayed: 0,
    selectedPair: null as { localType: string | null; remoteType: string | null; family: number } | null };
  readonly feedback = { video: new ReceiverFeedback(90000), audio: new ReceiverFeedback(48000) };
  constructor(private media: MediaSource, private allowed: () => boolean, private address: string,
    private catalog: readonly Tier[] = tiers, private endControllerSession?: (id: string) => Promise<void>) {}
  async handleSolicitOffer(request: WebRTCSolicitOfferRequest): Promise<WebRTCOffer> {
    const id = request.sessionIdentifier;
    this.counters.solicitations++;
    if (!this.allowed() || this.sessions.size >= 6 || this.sessions.has(id)) throw new Error('remote session unavailable');
    // Remote offers may be relayed over cellular. See the handset limits and
    // matching medium-tier offer in Scrypted #2152.
    const tier = this.catalog.filter(({ peakBitrate, width, height }) =>
      peakBitrate <= 800 && Math.max(width, height) <= 1280 && Math.min(width, height) <= 720)
      .sort((a, b) => b.peakBitrate - a.peakBitrate)[0];
    if (!tier) {
      this.counters.failures++;
      this.counters.lastFailure = 'no_cellular_video_tier';
      throw new Error('no compatible HomeKit remote video tier');
    }
    // Apple uses BUNDLE. max-compat gathers a second transport which this werift
    // revision loses when applying a bundled answer, leaving its UDP sockets open.
    let earlyDtls: ReturnType<typeof preserveEarlyClientHello> | undefined;
    const pc = new RTCPeerConnection({ bundlePolicy: 'max-bundle', iceServers: [], iceUseIpv6: false,
      // The candidate hook exposes source-aware UDP events before ICE selects a pair.
      iceFilterCandidatePair: pair => { earlyDtls?.observe(pair); return true; },
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
    video.sender.onRtcp.subscribe(packet => this.feedback.video.observe(packet, video.sender.ssrc));
    audio.sender.onRtcp.subscribe(packet => this.feedback.audio.observe(packet, audio.sender.ssrc));
    const startedAt = performance.now();
    const fail = (reason: string) => {
      if (session.closed) return;
      this.counters.failures++;
      this.counters.lastFailure = reason;
      void this.endSession(id).catch(() => console.error('HomeKit WebRTC session cleanup failed.'));
    };
    const transport = video.dtlsTransport;
    earlyDtls = preserveEarlyClientHello(transport, () => this.counters.earlyClientHellosReplayed++);
    const session: RemoteSession = { pc, earlyDtls, frame: new SecureVideoSFrame(true), closed: false, offerReady: false,
      timeout: setTimeout(() => fail('negotiation_timeout'), 60_000) };
    this.sessions.set(id, session);
    this.counters.lastState = pc.connectionState;
    this.counters.lastIceState = pc.iceConnectionState;
    this.counters.lastDtlsState = transport.state;
    this.counters.selectedPair = null;
    pc.iceConnectionStateChange.subscribe(state => {
      if (session.closed) return;
      this.counters.lastIceState = state;
      const selected = transport.iceTransport.getSelectedCandidatePair();
      if (selected) {
        const local = selected.local.candidate.split(/\s+/);
        const remote = selected.remote.candidate.split(/\s+/);
        const types = ['host', 'srflx', 'prflx', 'relay'];
        this.counters.selectedPair = {
          localType: types.includes(local[7]) ? local[7] : null,
          remoteType: types.includes(remote[7]) ? remote[7] : null,
          family: isIP(local[4] ?? ''),
        };
      }
    });
    transport.onStateChange.subscribe(state => {
      if (!session.closed) this.counters.lastDtlsState = state;
    });
    const candidates: NonNullable<WebRTCOffer['candidates']> = [];
    pc.onIceCandidate.subscribe(candidate => {
      if (!candidate) return;
      const value = candidate.toJSON();
      candidates.push({ candidate: value.candidate, sdpMid: value.sdpMid ?? undefined, sdpMLineIndex: value.sdpMLineIndex ?? undefined });
    });
    pc.connectionStateChange.subscribe(state => {
      if (!session.closed) this.counters.lastState = state;
      if (state === 'failed' || state === 'disconnected' || state === 'closed') fail('peer_' + state);
      if (state !== 'connected' || session.closed || session.feed) return;
      if (!this.allowed()) { fail('privacy_changed'); return; }
      this.counters.lastConnectTimeMs = Math.round(performance.now() - startedAt);
      clearTimeout(session.timeout);
      session.timeout = setTimeout(() => fail('session_timeout'), 10 * 60_000);
      const assembler = new HevcAccessUnitAssembler();
      const vc = session.frame.videoStream(video.sender.ssrc); const ac = session.frame.audioStream(audio.sender.ssrc);
      const vp = new SFrameRtpPacketizer({ ssrc: video.sender.ssrc, payloadType: video.sender.codec?.payloadType ?? 99, maxPayload: 1200 });
      const ap = new SFrameRtpPacketizer({ ssrc: audio.sender.ssrc, payloadType: audio.sender.codec?.payloadType ?? 110, maxPayload: 1200 });
      session.feed = new RtpFeed(this.media, tier);
      this.counters.started++;
      void session.feed.start(packet => {
        if (session.closed) return;
        const frame = assembler.push(packet);
        if (frame) for (const value of vp.packetize(vc.protectFrame(frame.data), frame.timestamp, true)) {
          // Match Scrypted's signaling-only RID interoperability behavior.
          // camera-webrtc.ts at fb679504ce2d8bc150bb504a7098826644627100.
          vtrack.writeRtp(value);
          this.counters.videoPackets++;
        }
      }, packet => {
        if (session.closed) return;
        for (const value of ap.packetize(ac.protectFrame(packet.payload), packet.header.timestamp, packet.header.marker)) {
          atrack.writeRtp(value);
          this.counters.audioPackets++;
        }
      }, () => fail('media_failed')).then(() => {
        if (!session.closed) this.counters.mediaReady++;
      }).catch(() => fail('media_failed'));
    });
    try {
      const offer = await pc.createOffer();
      this.requireActive(session);
      await pc.setLocalDescription(offer);
      this.requireActive(session);
      const sdpOffer = videoOffer(pc.localDescription!.sdp, tier);
      session.offerReady = true;
      this.counters.offers++;
      this.counters.lastVideoTier = { width: tier.width, height: tier.height, peakBitrate: tier.peakBitrate };
      this.counters.lastOfferTimeMs = Math.round(performance.now() - startedAt);
      this.counters.lastLocalCandidates = candidates.length;
      return { sdpOffer, candidates, sframe: session.frame.senderKey };
    } catch (error) {
      await this.signalingError(id, session, 'offer_failed');
      throw error;
    }
  }
  async handleProvideAnswer(request: WebRTCProvideAnswerRequest): Promise<void> {
    const id = request.sessionIdentifier;
    const session = this.sessions.get(id);
    if (!session?.offerReady || session.closed || session.signaling) throw new Error('unexpected remote answer');
    const answer = (async () => {
      this.requireActive(session);
      let sdp = request.sdpAnswer;
      if (request.candidates.length) {
        // Werift flushes queued candidates before connecting. Move SDP end
        // markers into that queue as otherwise BUNDLE can close ICE intake first.
        const endMarker = /^a=end-of-candidates[\t ]*(?:\r?\n|$)/gm;
        const complete = endMarker.test(sdp) || request.candidates.some(candidate => !candidate.candidate);
        sdp = sdp.replace(endMarker, '');
        for (const candidate of request.candidates) {
          if (!candidate.candidate) continue;
          // A candidate with no media hint belongs to the max-bundle transport.
          await session.pc.addIceCandidate(candidate.sdpMid === undefined && candidate.sdpMLineIndex === undefined
            ? { ...candidate, sdpMLineIndex: 0 } : candidate);
        }
        if (complete) await session.pc.addIceCandidate({ candidate: '' });
      }
      this.requireActive(session);
      await session.pc.setRemoteDescription({ type: 'answer', sdp });
      this.requireActive(session);
      this.counters.lastRemoteCandidates = session.pc.dtlsTransports[0].iceTransport.getRemoteCandidates().length;
      this.counters.answers++;
    })();
    session.signaling = answer;
    try { await answer; }
    catch (error) {
      await this.signalingError(id, session, 'answer_failed');
      throw error;
    }
  }
  async handleReoffer(request: WebRTCReofferRequest): Promise<WebRTCReofferAnswer> {
    const id = request.sessionIdentifier;
    const session = this.sessions.get(id);
    if (!session?.signaling || session.closed) throw new Error('unexpected remote reoffer');
    const reoffer = session.signaling.then(async () => {
      this.requireActive(session);
      await session.pc.setRemoteDescription({ type: 'offer', sdp: request.sdpOffer });
      this.requireActive(session);
      const answer = await session.pc.createAnswer();
      this.requireActive(session);
      await session.pc.setLocalDescription(answer);
      this.requireActive(session);
      this.counters.reoffers++;
      return { sdpAnswer: session.pc.localDescription!.sdp };
    });
    session.signaling = reoffer;
    try { return await reoffer; }
    catch (error) {
      await this.signalingError(id, session, 'reoffer_failed');
      throw error;
    }
  }
  private requireActive(session: RemoteSession): void {
    if (session.closed || !this.allowed()) throw new Error('remote session cancelled');
  }
  private async signalingError(id: string, session: RemoteSession, reason: string): Promise<void> {
    if (session.closed) return;
    this.counters.failures++;
    this.counters.lastFailure = reason;
    await this.endSession(id);
  }
  async handleUpdateSession(request: WebRTCUpdateSessionRequest): Promise<void> {
    const session = this.sessions.get(request.sessionIdentifier);
    if (!session || session.closed) throw new Error('unknown remote session');
    // Speaker advertisement is under hardware test; only the receive-key lifecycle
    // is handled here. Return audio is not decoded or played.
    session.frame.addReceiveKeys(request.receiveKeysToAdd); session.frame.removeReceiveKeys(request.receiveKIDsToRemove);
  }
  async handleEndSession(id: string): Promise<void> {
    const session = this.sessions.get(id);
    if (!session) return;
    if (session.stopping) return session.stopping;
    if (session.closed) return; // close() can emit synchronously before the promise is assigned.
    session.closed = true; clearTimeout(session.timeout);
    session.earlyDtls.stop();
    session.stopping = (async () => {
      try { await session.feed?.stop(); }
      finally {
        try { await session.pc.close(); }
        finally { this.sessions.delete(id); }
      }
    })();
    return session.stopping;
  }
  private async endSession(id: string): Promise<void> {
    // Start delegate teardown before notifying HAP. Its end hook calls back
    // into handleEndSession, which joins the same teardown without recursing.
    const teardown = this.handleEndSession(id);
    await Promise.all([teardown, this.endControllerSession?.(id)]);
  }
  async close(): Promise<void> { await Promise.all([...this.sessions.keys()].map(id => this.endSession(id))); }
}
