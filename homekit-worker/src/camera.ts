import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import { dirname, isAbsolute, join } from 'node:path';
import { isIPv4 } from 'node:net';
import { networkInterfaces } from 'node:os';
import { fstatSync } from 'node:fs';

import { Accessory, Categories, Characteristic, HAPStorage, Service, SecureVideoController,
  StreamTierVideoCodec, StreamTierAudioSampleRate, StreamTierAudioBitDepth, MediaContainerType,
  VideoCodecType, H264Level, H264Profile, AudioRecordingCodecType, AudioBitrate, AudioRecordingSamplerate, uuid } from '@homebridge/hap-nodejs';
import { controlServer, identity, privateDirectory } from './control.js';
import { LabRecording } from './fragments.js';
import { LocalStreams, RemoteStreams } from './transports.js';
import { qualifyEncoding, selectEncoding, streamProfile, EncodingUnavailableError, type VideoEncoding } from './encoding.js';
import { CameraMedia, type MediaAdmission } from './camera-media.js';

async function main(): Promise<void> {
  process.umask(0o077);
  const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const iterator = lines[Symbol.asyncIterator]();
  const first = await iterator.next();
  if (first.done || first.value.length > 16384) throw new Error('invalid supervisor configuration');
  const config = JSON.parse(first.value) as {
    version: number; state: string; bind: string; port: number; name: string;
    url: string; audio: boolean; ffmpeg: string; transport: 'tcp' | 'udp'; allowSoftwareEncoding: boolean; ownerFd: number;
  };
  if (process.versions.node.split('.')[0] !== '24' || config.version !== 1) throw new Error('Node.js 24 and protocol v1 required');
  // Retain the inherited flock descriptor for this process lifetime. A new
  // supervisor cannot replace our sockets while we drain after parent EOF.
  if (!Number.isInteger(config.ownerFd) || config.ownerFd < 3 || !fstatSync(config.ownerFd).isFile()) throw new Error('missing owner lock');
  const state = config.state, address = config.bind, port = config.port;
  if (!isAbsolute(state) || !isIPv4(address) || !Object.values(networkInterfaces()).flat().some(info => info?.address === address)) throw new Error('invalid private state or LAN bind');
  if (!Number.isInteger(port) || port < 1024 || port > 65535 || !/^https?:|^rtsps?:/i.test(config.url)) throw new Error('invalid camera configuration');
  await privateDirectory(state);
  const waiting = new Map<number, { resolve(): void; reject(error: Error): void }>();
  let serial = 0;
  const admission: MediaAdmission = (method, token, detail) => new Promise((resolve, reject) => {
    const id = ++serial;
    const timer = setTimeout(() => { waiting.delete(id); reject(new Error('supervisor unavailable')); }, 3000);
    waiting.set(id, { resolve: () => { clearTimeout(timer); resolve(); }, reject: error => { clearTimeout(timer); reject(error); } });
    process.stdout.write(JSON.stringify({ version: 1, id, method, token, ...detail }) + '\n');
  });
  let controller: SecureVideoController | undefined;
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
    qualifying = selectEncoding(config.allowSoftwareEncoding,
      () => qualifyEncoding(config.ffmpeg, state, admission, qualificationAbort.signal, { backend: 'qsv', device: '/dev/dri/renderD128' }),
      () => qualifyEncoding(config.ffmpeg, state, admission, qualificationAbort.signal, { backend: 'software' }));
    const encoding = await qualifying;
    if (closing) throw new Error('camera stopping');
    let frame = { width: 1920, height: 1080 };
    try { frame = await probeFrame(config.ffmpeg, config.url, config.transport); }
    catch { console.error('HomeKit camera frame probe failed; using 16:9.'); }
    const profile = streamProfile(frame.width, frame.height);
    media = new CameraMedia(config.url, config.audio, config.ffmpeg, config.transport, admission,
      () => !closing && !!controller?.homeKitCameraActive, state, encoding, profile.tiers);
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
    remote = new RemoteStreams(media, () => !closing && !!controller?.streamingAllowed(controller.webrtcStreamManagementService), address!, profile.tiers);
    recording = new LabRecording(media, cameraAllowed, profile.tiers);
    controller = new SecureVideoController({
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
          resolutions: profile.resolutions },
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
    const setMotion = (active: boolean) => {
      const allowed = cameraAllowed() && !!controller!.motionService?.getCharacteristic(Characteristic.MotionEnabled).value;
      controller!.setMotionDetected(active && allowed); recording!.setMotion(active && allowed);
    };
    controller.on('operating-mode-changed', () => {
      if (!controller!.streamingAllowed(controller!.multiTierStreamManagementService)) void local!.close();
      if (!controller!.streamingAllowed(controller!.webrtcStreamManagementService)) void remote!.close();
      if (!cameraAllowed()) { setMotion(false); void media!.cancelSnapshots(); }
      recording!.setAudio(controller!.recordingAudioActive); recording!.reconcile();
    });
    controller.recordingManagementService!.getCharacteristic(Characteristic.RecordingAudioActive).on('change', () => {
      recording!.setAudio(controller!.recordingAudioActive);
    });
    producerRetry = setInterval(() => recording!.reconcile(), 5000);
    control = await controlServer(join(state, 'control.sock'), (method, params) => {
      if (method === 'status') return { schemaVersion: 1, mode: 'camera', acceptance: 'experimental',
        paired: !!accessory!._accessoryInfo?.paired(), childProcesses: media!.children.size,
        localSessions: local!.sessions.size, remoteSessions: remote!.sessions.size,
        homeKitActive: cameraAllowed(), recording: recording!.status(), media: { ...media!.counters, lastError: media!.lastError },
        encoder: media!.encoderStatus,
        localMedia: { sent: local!.sent, feedback: { video: local!.feedback.video.stats, audio: local!.feedback.audio.stats } },
        zonesBlockMotion: !!(controller!.motionZones.active && controller!.motionZones.zones),
      };
      if (method === 'pairing') return { setupCode: saved.pincode, setupUri: accessory!.setupURI() };
      if (method === 'motion') {
        const active = (params as { active?: unknown })?.active;
        if (typeof active !== 'boolean') throw new Error('invalid motion state');
        clearTimeout(motionTimer);
        // These events have no HomeKit spatial coordinates. Fail closed for
        // Home activity zones rather than triggering outside a selected zone.
        setMotion(active && !(controller!.motionZones.active && controller!.motionZones.zones));
        motionTimer = setTimeout(() => setMotion(false), 5000);
        return { accepted: true };
      }
      throw new Error('unknown control method');
    });
    await accessory.publish({ username: saved.username, pincode: saved.pincode, category: Categories.IP_CAMERA, port, bind: address });
    console.error('SurvNG HomeKit camera ready.');
  } catch (error) {
    if (!closing) exitStatus = error instanceof EncodingUnavailableError ? 42 : 1;
    await stop(); throw error;
  }
}

async function probeFrame(ffmpeg: string, url: string, transport: 'tcp' | 'udp'): Promise<{ width: number; height: number }> {
  const args = ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,sample_aspect_ratio', '-of', 'json'];
  if (/^rtsps?:/i.test(url)) args.unshift('-rtsp_transport', transport);
  const child = spawn(join(dirname(ffmpeg), 'ffprobe'), [...args, url], { stdio: ['ignore', 'pipe', 'ignore'] });
  const chunks: Buffer[] = [];
  child.stdout!.on('data', (chunk: Buffer) => { if (chunks.reduce((sum, item) => sum + item.length, 0) < 65536) chunks.push(chunk); });
  const timer = setTimeout(() => child.kill('SIGKILL'), 15000);
  try {
    const [code] = await once(child, 'close');
    if (code !== 0) throw new Error('camera frame probe failed');
    const stream = JSON.parse(Buffer.concat(chunks).toString('utf8'))?.streams?.[0];
    const width = Number(stream?.width), height = Number(stream?.height);
    if (!Number.isInteger(width) || !Number.isInteger(height)) throw new Error('camera frame probe failed');
    const sar = /^(\d+):(\d+)$/.exec(String(stream?.sample_aspect_ratio ?? ''));
    if (!sar || !Number(sar[1]) || !Number(sar[2])) return { width, height };
    return { width: Math.max(2, Math.round(width * Number(sar[1]) / Number(sar[2]) / 2) * 2), height };
  } finally { clearTimeout(timer); }
}

main().catch(error => { console.error('SurvNG HomeKit camera stopped with an error.'); process.exitCode = error instanceof EncodingUnavailableError ? 42 : 1; });
