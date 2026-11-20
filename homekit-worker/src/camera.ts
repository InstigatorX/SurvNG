import { createInterface } from 'node:readline';
import { isAbsolute, join } from 'node:path';
import { isIPv4 } from 'node:net';
import { networkInterfaces } from 'node:os';
import { fstatSync } from 'node:fs';

import { Accessory, Categories, Characteristic, HAPStorage, Service,
  StreamTierVideoCodec, StreamTierAudioSampleRate, StreamTierAudioBitDepth, MediaContainerType,
  VideoCodecType, H264Level, H264Profile, AudioRecordingCodecType, AudioBitrate, AudioRecordingSamplerate, uuid } from '@homebridge/hap-nodejs';
import { controlServer, identity, privateDirectory } from './control.js';
import { HomeKitMotionController, motionZonesBlockUnlocatedMotion } from './homekit-motion.js';
import { LabRecording } from './fragments.js';
import { serviceStatus } from './diagnostics.js';
import { LocalStreams, RemoteStreams } from './transports.js';
import { probeFrame, qualifyEncoding, selectEncoding, streamProfile, recordingProfile, EncodingUnavailableError, type VideoEncoding } from './encoding.js';
import type { Tier } from './media.js';
import { CameraMedia, type MediaAdmission } from './camera-media.js';

async function main(): Promise<void> {
  process.umask(0o077);
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const iterator = lines[Symbol.asyncIterator]();
  const first = await iterator.next();
  if (first.done || first.value.length > 16384) throw new Error('invalid supervisor configuration');
  const config = JSON.parse(first.value) as {
    version: number; state: string; bind: string; port: number; name: string;
    url?: string; audio: boolean; ffmpeg: string; transport: 'tcp' | 'udp'; allowSoftwareEncoding: boolean; ownerFd: number;
    remote?: boolean; feeds?: string; profile?: { width: number; height: number; tiers: Tier[]; resolutions: [number, number, number][] };
    encoding?: VideoEncoding; readrateCatchup?: boolean;
  };
  if (process.versions.node.split('.')[0] !== '24' || config.version !== 1) throw new Error('Node.js 24 and protocol v1 required');
  // Retain the inherited flock descriptor for this process lifetime. A new
  // supervisor cannot replace our sockets while we drain after parent EOF.
  if (!Number.isInteger(config.ownerFd) || config.ownerFd < 3 || !fstatSync(config.ownerFd).isFile()) throw new Error('missing owner lock');
  const state = config.state, address = config.bind, port = config.port;
  if (!isAbsolute(state) || !isIPv4(address) || !Object.values(networkInterfaces()).flat().some(info => info?.address === address)) throw new Error('invalid private state or LAN bind');
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('invalid camera configuration');
  if (config.remote) {
    if (!config.feeds || !isAbsolute(config.feeds) || !config.profile?.tiers?.length) throw new Error('invalid remote encoder');
  } else if (!config.url || !/^https?:|^rtsps?:/i.test(config.url)) throw new Error('invalid camera configuration');
  await privateDirectory(state);
  const waiting = new Map<number, { resolve(): void; reject(error: Error): void }>();
  let serial = 0;
  const admission: MediaAdmission = (method, token, detail) => new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => { waiting.delete(id); reject(new Error('supervisor unavailable')); }, method === 'snapshot' ? 15000 : 3000);
    waiting.set(id, { resolve: () => { clearTimeout(timer); resolve(); }, reject: error => { clearTimeout(timer); reject(error); } });
    process.stdout.write(JSON.stringify({ version: 1, id, method, token, ...detail }) + '\n');
  });
  let controller: HomeKitMotionController | undefined;
  let media: CameraMedia | undefined;
  let qualifying: Promise<VideoEncoding> | undefined;
  const qualificationAbort = new AbortController();
  let stopping: Promise<void> | undefined;
  let accessory: Accessory | undefined;
  let local: LocalStreams | undefined;
  let remote: RemoteStreams | undefined;
  let recording: LabRecording | undefined;
  let control: Awaited<ReturnType<typeof controlServer>> | undefined;
  let producerRetry: NodeJS.Timeout | undefined;
  let motionTimer: NodeJS.Timeout | undefined;
  let closing = false;
  let exitStatus = 0;
  const stop = (): Promise<void> => stopping ??= (async () => {
    closing = true; qualificationAbort.abort(); clearTimeout(motionTimer); clearInterval(producerRetry);
    const results = await Promise.allSettled([control?.close(), accessory?.unpublish(), local?.close(), remote?.close(), recording?.close()]);
    await qualifying?.catch(() => {});
    await media?.stop(); lines.close();
    if (results.some(result => result.status === 'rejected')) throw new Error('camera shutdown incomplete');
  })();
  const signalStop = () => { void stop().then(() => process.exit(exitStatus), () => process.exit(1)); };
  process.stdout.on('error', signalStop);
  process.once('SIGTERM', signalStop); process.once('SIGINT', signalStop);
  void (async () => {
    for await (const line of { [Symbol.asyncIterator]: () => iterator }) {
      try {
        const response = JSON.parse(line);
        const request = waiting.get(response.id);
        waiting.delete(response.id);
        if (response.ok) request?.resolve(); else request?.reject(new Error('media admission rejected'));
      } catch { signalStop(); return; }
    }
    signalStop();
  })();
  try {
    let encoding: VideoEncoding;
    let profile: { width: number; height: number; tiers: Tier[]; resolutions: [number, number, number][] };
    if (config.remote) {
      encoding = config.encoding?.backend === 'qsv' || config.encoding?.backend === 'software' ? config.encoding : { backend: 'software' };
      profile = config.profile!;
    } else {
      qualifying = selectEncoding(config.allowSoftwareEncoding,
        () => qualifyEncoding(config.ffmpeg, state, admission, qualificationAbort.signal, { backend: 'qsv', device: '/dev/dri/renderD128' }),
        () => qualifyEncoding(config.ffmpeg, state, admission, qualificationAbort.signal, { backend: 'software' }));
      encoding = await qualifying;
      if (closing) throw new Error('camera stopping');
      let frame = { width: 1920, height: 1080 };
      try { frame = await probeFrame(config.ffmpeg, config.url!, config.transport, qualificationAbort.signal); }
      catch { console.error('HomeKit camera frame probe failed; using 16:9.'); }
      qualificationAbort.signal.throwIfAborted();
      profile = streamProfile(frame.width, frame.height);
    }
    media = new CameraMedia(config.url || 'http://127.0.0.1/homekit-remote', config.audio, config.ffmpeg, config.transport, admission,
      () => !closing && !!controller?.homeKitCameraActive, state, encoding, profile.tiers, config.remote ? config.feeds : '',
      config.readrateCatchup ?? true);
    const saved = await identity(state);
    const persist = join(state, 'hap'); await privateDirectory(persist);
    HAPStorage.setCustomStoragePath(persist);
    accessory = new Accessory(config.name, uuid.generate(saved.id));
    accessory.getService(Service.AccessoryInformation)!
      .setCharacteristic(Characteristic.Manufacturer, 'SurvNG')
      .setCharacteristic(Characteristic.Model, 'SurvNG HKSV3 camera')
      .setCharacteristic(Characteristic.SerialNumber, saved.id)
      .setCharacteristic(Characteristic.FirmwareRevision, '0.1.0');
    const cameraAllowed = () => !closing && !!controller?.homeKitCameraActive;
    local = new LocalStreams(media, () => !closing && !!controller?.streamingAllowed(controller.multiTierStreamManagementService), profile.tiers);
    remote = new RemoteStreams(media, () => !closing && !!controller?.streamingAllowed(controller.webrtcStreamManagementService),
      address!, profile.tiers, id => controller!.endWebRTCSession(id));
    const hksv = recordingProfile(profile, process.env.SURVNG_HOMEKIT_SHARED_RECORDING_TIER === '1');
    recording = new LabRecording(media, cameraAllowed, hksv.tiers);
    controller = new HomeKitMotionController({
      sensor: { uuid: uuid.generate(`${saved.id}:sensor`), width: profile.width, height: profile.height },
      video: { codec: StreamTierVideoCodec.H265, payloadType: 99, tiers: profile.tiers },
      // Apple compatibility requires Speaker advertisement; talkback remains unavailable.
      audio: { payloadType: 110, twoWayAudio: true, tier: { identifier: 1, targetAverageBitrate: 24000,
        sampleRate: StreamTierAudioSampleRate.KHZ_48, bitDepth: StreamTierAudioBitDepth.BITS_16, packetTime: 20, channels: 1 } },
      rtp: { delegate: local }, webrtc: { delegate: remote, maxSessions: 6 },
      recording: { delegate: recording, options: {
        prebufferLength: 4000,
        mediaContainerConfiguration: { type: MediaContainerType.FRAGMENTED_MP4, fragmentLength: 4000 },
        // The pinned HKSV3 controller uses the legacy HDS configuration schema while carrying HEVC.
        // Home will not select a profile that lacks 15 fps or uses a fragment length other than 4 seconds.
        video: { type: VideoCodecType.H264, parameters: { levels: [H264Level.LEVEL4_0], profiles: [H264Profile.MAIN] },
          resolutions: hksv.resolutions },
        audio: { codecs: [{ type: AudioRecordingCodecType.AAC_LC, bitrateMode: AudioBitrate.VARIABLE,
          samplerate: AudioRecordingSamplerate.KHZ_32, audioChannels: 1 }] },
      } },
      snapshot: async () => {
        if (!cameraAllowed()) throw new Error('HomeKit privacy enabled');
        const jpeg = await media!.snapshot();
        if (!cameraAllowed()) throw new Error('HomeKit privacy enabled');
        return jpeg;
      },
    });
    accessory.configureController(controller);
    // SurvNG does not report its own local viewers as third-party Home activity.
    controller.globalOperatingModeService?.setCharacteristic(
      Characteristic.ThirdPartyCameraActive, Characteristic.ThirdPartyCameraActive.OFF);
    const motionEnabled = () => !!controller!.motionService?.getCharacteristic(Characteristic.MotionEnabled).value;
    const zonesBlockMotion = () => motionZonesBlockUnlocatedMotion(controller!.motionZones);
    const motionPermitted = () => cameraAllowed() && motionEnabled() && !zonesBlockMotion();
    const publishMotion = () => controller!.setMotionDetected(motionPermitted() && recording!.reportingMotion());
    const setMotion = (active: boolean) => { recording!.setMotion(active && motionPermitted()); publishMotion(); };
    recording!.watch(publishMotion);
    controller.on('operating-mode-changed', () => {
      if (!controller!.streamingAllowed(controller!.multiTierStreamManagementService)) void local!.close();
      if (!controller!.streamingAllowed(controller!.webrtcStreamManagementService)) {
        void remote!.close().catch(() => console.error('HomeKit WebRTC session cleanup failed.'));
      }
      if (!cameraAllowed()) { setMotion(false); void media!.cancelSnapshots(); }
      recording!.setAudio(controller!.recordingAudioActive); recording!.reconcile();
    });
    controller.recordingManagementService!.getCharacteristic(Characteristic.RecordingAudioActive).on('change', () => {
      recording!.setAudio(controller!.recordingAudioActive);
    });
    producerRetry = setInterval(() => recording!.reconcile(), 5000);
    // Hub activity, as in the protocol lab: allowlisted counters and per-characteristic
    // tallies only. Never retain characteristic values, pairing material, SDP, or addresses.
    const requests = { accessories: 0, reads: 0, writes: 0, resources: 0, lastAt: 0 };
    const characteristics: Record<string, { reads: number; writes: number; subscriptions: number }> = {};
    control = await controlServer(join(state, 'control.sock'), (method, params) => {
      if (method === 'status') return { schemaVersion: 1, mode: 'camera', acceptance: 'experimental',
        paired: !!accessory!._accessoryInfo?.paired(), childProcesses: media!.children.size,
        localSessions: local!.sessions.size, remoteSessions: remote!.sessions.size, remoteControllerSessions: controller!.activeWebRTCSessions.length,
        homeKitActive: cameraAllowed(), recording: { ...recording!.status(), sharedTier: hksv.sharedTier?.identifier ?? null }, media: { ...media!.counters, lastError: media!.lastError },
        encoder: media!.encoderStatus,
        localMedia: { sent: local!.sent, ...local!.counters, negotiation: local!.negotiation ?? null,
          feedback: { video: local!.feedback.video.stats, audio: local!.feedback.audio.stats } },
        remoteMedia: { ...remote!.counters, videoFeedback: remote!.feedback.video.stats, audioFeedback: remote!.feedback.audio.stats },
        motion: { enabled: motionEnabled(), permitted: motionPermitted(), detected: motionPermitted() && recording!.reportingMotion() },
        zonesBlockMotion: zonesBlockMotion(),
        hub: { ...requests, lastAt: undefined, lastRequestAgeSeconds: requests.lastAt ? Math.round((Date.now() - requests.lastAt) / 1000) : null,
          characteristics },
        services: {
          globalOperatingMode: serviceStatus(controller!.globalOperatingModeService),
          recordingOperatingMode: serviceStatus(accessory!.getService(Service.CameraOperatingMode)),
          localStreaming: serviceStatus(controller!.multiTierStreamManagementService),
          remoteStreaming: serviceStatus(controller!.webrtcStreamManagementService),
        },
        localAllowed: controller!.streamingAllowed(controller!.multiTierStreamManagementService),
        remoteAllowed: controller!.streamingAllowed(controller!.webrtcStreamManagementService),
      };
      if (method === 'pairing') return { setupCode: saved.pincode, setupUri: accessory!.setupURI() };
      if (method === 'motion') {
        const active = (params as { active?: unknown })?.active;
        if (typeof active !== 'boolean') throw new Error('invalid motion state');
        clearTimeout(motionTimer);
        setMotion(active);
        motionTimer = setTimeout(() => setMotion(false), 5000);
        return { accepted: true };
      }
      throw new Error('unknown control method');
    });
    await accessory.publish({ username: saved.username, pincode: saved.pincode, category: Categories.IP_CAMERA, port, bind: address });
    const countCharacteristics = (items: { aid: number; iid: number; value?: unknown; ev?: boolean }[], operation: 'reads' | 'writes') => {
      for (const item of items) {
        if (item.aid !== 1) continue;
        const characteristic = accessory!.services.flatMap(service => service.characteristics)
          .find(candidate => candidate.iid === item.iid);
        if (!characteristic) continue;
        const count = characteristics[characteristic.displayName] ??= { reads: 0, writes: 0, subscriptions: 0 };
        if (operation === 'reads' || Object.hasOwn(item, 'value')) count[operation]++;
        if (operation === 'writes' && item.ev !== undefined) count.subscriptions++;
      }
    };
    accessory._server!.on('accessories', () => { requests.accessories++; requests.lastAt = Date.now(); });
    accessory._server!.on('get-characteristics', (_connection, request) => {
      requests.reads++; requests.lastAt = Date.now(); countCharacteristics(request.ids, 'reads');
    });
    accessory._server!.on('set-characteristics', (_connection, request) => {
      requests.writes++; requests.lastAt = Date.now(); countCharacteristics(request.characteristics, 'writes');
    });
    accessory._server!.on('request-resource', () => { requests.resources++; requests.lastAt = Date.now(); });
    console.error('SurvNG HomeKit camera ready.');
  } catch (error) {
    if (!closing) exitStatus = error instanceof EncodingUnavailableError ? 42 : 1;
    await stop(); throw error;
  }
}

main().catch(error => { console.error('SurvNG HomeKit camera stopped with an error.'); process.exitCode = error instanceof EncodingUnavailableError ? 42 : 1; });
