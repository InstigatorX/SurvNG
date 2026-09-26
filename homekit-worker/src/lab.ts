import { parseArgs } from 'node:util';
import { isAbsolute, join } from 'node:path';
import { isIPv4 } from 'node:net';
import { networkInterfaces } from 'node:os';
import { open, unlink } from 'node:fs/promises';
import { Accessory, Categories, Characteristic, HAPStorage, Service, SecureVideoController,
  StreamTierVideoCodec, StreamTierAudioSampleRate, StreamTierAudioBitDepth, MediaContainerType,
  VideoCodecType, H264Level, H264Profile, AudioRecordingCodecType, AudioBitrate, AudioRecordingSamplerate, uuid } from '@homebridge/hap-nodejs';
import { controlServer, identity, privateDirectory } from './control.js';
import { SyntheticMedia, tiers } from './media.js';
import { LabRecording } from './fragments.js';
import { LocalStreams, RemoteStreams } from './transports.js';

async function main(): Promise<void> {
  const { values } = parseArgs({ options: {
    'state-dir': { type: 'string' }, bind: { type: 'string' }, port: { type: 'string', default: '51826' },
    ffmpeg: { type: 'string', default: 'ffmpeg' }, prepare: { type: 'boolean', default: false },
    'allow-synthetic-encoding': { type: 'boolean', default: false },
  } });
  if (process.versions.node.split('.')[0] !== '24') throw new Error('Node.js 24 required');
  const state = values['state-dir'];
  if (!state || !isAbsolute(state)) throw new Error('--state-dir must be an absolute private directory');
  if (!values['allow-synthetic-encoding']) throw new Error('synthetic CPU encoding requires --allow-synthetic-encoding');
  const address = values.bind;
  if (!values.prepare && (!address || !isIPv4(address) || !Object.values(networkInterfaces()).flat().some(info => info?.address === address))) {
    throw new Error('--bind must be this host’s LAN IPv4 address');
  }
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 1024 || port > 65535) throw new Error('invalid HAP port');
  process.umask(0o077);
  await privateDirectory(state);
  const lockPath = join(state, 'lab.lock');
  const lock = await open(lockPath, 'wx', 0o600);
  await lock.writeFile(String(process.pid)); await lock.close();
  const media = new SyntheticMedia(join(state, 'fixtures'), values.ffmpeg);
  let stopping: Promise<void> | undefined;
  let accessory: Accessory | undefined;
  let local: LocalStreams | undefined;
  let remote: RemoteStreams | undefined;
  let recording: LabRecording | undefined;
  let control: Awaited<ReturnType<typeof controlServer>> | undefined;
  let motionTimer: NodeJS.Timeout | undefined;
  let closing = false;
  const stop = (): Promise<void> => stopping ??= (async () => {
    closing = true; clearTimeout(motionTimer);
    const results = await Promise.allSettled([control?.close(), accessory?.unpublish(), local?.close(), remote?.close(), recording?.close()]);
    await media.stop(); await unlink(lockPath);
    if (results.some(result => result.status === 'rejected')) throw new Error('lab shutdown incomplete');
  })();
  const signalStop = () => { void stop().then(() => process.exit(0), () => process.exit(1)); };
  process.once('SIGTERM', signalStop); process.once('SIGINT', signalStop);
  try {
    await media.prepare();
    if (values.prepare) { await stop(); return; }
    const saved = await identity(state);
    const persist = join(state, 'hap'); await privateDirectory(persist);
    HAPStorage.setCustomStoragePath(persist);
    accessory = new Accessory('SurvNG HKSV3 Lab', uuid.generate(saved.id));
    accessory.getService(Service.AccessoryInformation)!
      .setCharacteristic(Characteristic.Manufacturer, 'SurvNG')
      .setCharacteristic(Characteristic.Model, 'Synthetic HKSV3 acceptance camera')
      .setCharacteristic(Characteristic.SerialNumber, saved.id)
      .setCharacteristic(Characteristic.FirmwareRevision, '0.1.0');
    let controller: SecureVideoController | undefined;
    const cameraAllowed = () => !closing && !!controller?.homeKitCameraActive;
    local = new LocalStreams(media, () => !closing && !!controller?.streamingAllowed(controller.multiTierStreamManagementService));
    remote = new RemoteStreams(media, () => !closing && !!controller?.streamingAllowed(controller.webrtcStreamManagementService), address!);
    recording = new LabRecording(media, cameraAllowed);
    const snapshots = { requested: 0, completed: 0, failed: 0 };
    const requests = { accessories: 0, reads: 0, writes: 0, resources: 0 };
    const characteristics: Record<string, { reads: number; writes: number; subscriptions: number }> = {};
    controller = new SecureVideoController({
      sensor: { uuid: uuid.generate(`${saved.id}:sensor`), width: 1920, height: 1080 },
      video: { codec: StreamTierVideoCodec.H265, payloadType: 99, tiers },
      audio: { payloadType: 110, twoWayAudio: false, tier: { identifier: 1, targetAverageBitrate: 24000,
        sampleRate: StreamTierAudioSampleRate.KHZ_48, bitDepth: StreamTierAudioBitDepth.BITS_16, packetTime: 20, channels: 1 } },
      rtp: { delegate: local }, webrtc: { delegate: remote, maxSessions: 6 },
      recording: { delegate: recording, options: {
        prebufferLength: 4000,
        mediaContainerConfiguration: { type: MediaContainerType.FRAGMENTED_MP4, fragmentLength: 2000 },
        // The pinned HKSV3 controller uses the legacy HDS configuration schema while carrying HEVC.
        video: { type: VideoCodecType.H264, parameters: { levels: [H264Level.LEVEL4_0], profiles: [H264Profile.MAIN] },
          resolutions: tiers.slice(0, 2).map(tier => [tier.width, tier.height, tier.frameRate]) },
        audio: { codecs: [{ type: AudioRecordingCodecType.AAC_LC, bitrateMode: AudioBitrate.VARIABLE,
          samplerate: AudioRecordingSamplerate.KHZ_32, audioChannels: 1 }] },
      } },
      snapshot: async () => {
        snapshots.requested++;
        try {
          if (!cameraAllowed()) throw new Error('HomeKit privacy enabled');
          const jpeg = await media.snapshot(); snapshots.completed++; return jpeg;
        } catch (error) { snapshots.failed++; throw error; }
      },
    });
    accessory.configureController(controller);
    const setMotion = (active: boolean) => {
      const allowed = cameraAllowed() && !!controller!.motionService?.getCharacteristic(Characteristic.MotionEnabled).value;
      controller!.setMotionDetected(active && allowed); recording!.setMotion(active && allowed);
    };
    controller.on('operating-mode-changed', () => {
      if (!controller!.streamingAllowed(controller!.multiTierStreamManagementService)) void local!.close();
      if (!controller!.streamingAllowed(controller!.webrtcStreamManagementService)) void remote!.close();
      if (!cameraAllowed()) setMotion(false);
      recording!.setAudio(controller!.recordingAudioActive); recording!.reconcile();
    });
    controller.recordingManagementService!.getCharacteristic(Characteristic.RecordingAudioActive).on('change', () => {
      recording!.setAudio(controller!.recordingAudioActive);
    });
    control = await controlServer(join(state, 'control.sock'), (method, params) => {
      if (method === 'status') return { schemaVersion: 1, mode: 'synthetic-lab', acceptance: 'unverified',
        paired: !!accessory!._accessoryInfo?.paired(),
        localSessions: local!.sessions.size, remoteSessions: remote!.sessions.size,
        local: local!.counters, remote: remote!.counters, snapshots, requests, characteristics,
        localAllowed: controller!.streamingAllowed(controller!.multiTierStreamManagementService),
        remoteAllowed: controller!.streamingAllowed(controller!.webrtcStreamManagementService),
        childProcesses: media.children.size, homeKitActive: cameraAllowed(), recording: recording!.status() };
      if (method === 'pairing') return { setupCode: saved.pincode, setupUri: accessory!.setupURI() };
      if (method === 'motion') {
        const seconds = (params as { seconds?: unknown } | undefined)?.seconds ?? 10;
        if (typeof seconds !== 'number' || seconds < 1 || seconds > 60 || !cameraAllowed()) throw new Error('invalid motion request');
        if (controller!.motionZones.active && controller!.motionZones.zones) throw new Error('disable zones for unlocated synthetic motion');
        clearTimeout(motionTimer); setMotion(true); motionTimer = setTimeout(() => setMotion(false), seconds * 1000);
        return { accepted: true };
      }
      throw new Error('unknown control method');
    });
    await accessory.publish({ username: saved.username, pincode: saved.pincode, category: Categories.IP_CAMERA, port, bind: address });
    // Allowlisted counters only: never retain characteristic values, pairing material, or SDP.
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
    accessory._server!.on('accessories', () => { requests.accessories++; });
    accessory._server!.on('get-characteristics', (_connection, request) => {
      requests.reads++; countCharacteristics(request.ids, 'reads');
    });
    accessory._server!.on('set-characteristics', (_connection, request) => {
      requests.writes++; countCharacteristics(request.characteristics, 'writes');
    });
    accessory._server!.on('request-resource', () => { requests.resources++; });
    console.log('Synthetic HKSV3 lab ready. Use the private control client to retrieve pairing information.');
  } catch (error) { await stop(); throw error; }
}

main().catch(error => { console.error(error instanceof Error ? error.message : 'Lab failed'); process.exitCode = 1; });
