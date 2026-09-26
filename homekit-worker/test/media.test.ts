import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { setTimeout as delay } from 'node:timers/promises';
import { RtpPacket, SrtpSession, RTCPeerConnection, RTCRtpCodecParameters, RTCRtpHeaderExtensionParameters } from 'werift';
import { SyntheticMedia, tiers, udp } from '../src/media.js';
import { LabRecording, boxes, videoClock, decodeTime } from '../src/fragments.js';
import { LocalStreams, RemoteStreams, STREAM_ID_URI } from '../src/transports.js';
import { SFrameCipherSuite, SFrameReceiver } from '../src/vendor/sframe.js';
import { SFrameRtpDepacketizer } from '../src/vendor/sframeRtp.js';
import type { CameraRecordingConfiguration } from '@homebridge/hap-nodejs';

const execute = promisify(execFile);
async function until(predicate: () => boolean, milliseconds = 15000): Promise<void> {
  const deadline = Date.now() + milliseconds;
  while (!predicate()) { if (Date.now() > deadline) throw new Error('condition timed out'); await delay(50); }
}
test('real HEVC fixtures, fMP4 prebuffer, local SRTP, and remote encrypted media', {
  skip: process.env.HKSV_MEDIA_TESTS !== '1', timeout: 120000,
}, async t => {
  const root = await mkdtemp(join(tmpdir(), 'survng-hksv-media-'));
  const media = new SyntheticMedia(root);
  let allowed = true;
  const local = new LocalStreams(media, () => allowed);
  const remote = new RemoteStreams(media, () => allowed, '127.0.0.1');
  const recording = new LabRecording(media, () => allowed);
  try {
    await media.prepare();
    await t.test('every advertised tier is HEVC with matching dimensions and cadence', async () => {
      for (const tier of tiers) {
        const result = await execute('ffprobe', ['-v', 'error', '-show_streams', '-of', 'json', media.path(tier)]);
        const streams = JSON.parse(result.stdout).streams;
        const video = streams.find((stream: { codec_type: string }) => stream.codec_type === 'video');
        assert.equal(video.codec_name, 'hevc'); assert.equal(video.codec_tag_string, 'hvc1');
        assert.equal(video.width, tier.width); assert.equal(video.height, tier.height);
        assert.equal(video.r_frame_rate, `${tier.frameRate}/1`);
        assert.equal(streams.find((stream: { codec_type: string }) => stream.codec_type === 'audio').codec_name, 'aac');
      }
      const jpeg = await media.snapshot(); assert.equal(jpeg.readUInt16BE(0), 0xffd8);
    });
    await t.test('HDS pre-roll has monotonic decode time, prft, and playable HEVC/AAC', async () => {
      recording.updateRecordingConfiguration({
        prebufferLength: 4000, eventTriggerTypes: [1], mediaContainerConfiguration: { type: 0, fragmentLength: 2000 },
        videoCodec: { type: 0, parameters: { profile: 1, level: 2, bitRate: 1700, iFrameInterval: 2000 }, resolution: [1920, 1080, 30] },
        audioCodec: { type: 0, audioChannels: 1, bitrateMode: 0, samplerate: 3, bitrate: 32 },
      } as CameraRecordingConfiguration);
      recording.updateRecordingActive(true);
      await until(() => recording.ring.coverage >= 4);
      recording.setMotion(true);
      const stream = recording.handleRecordingStreamRequest(1);
      const init = (await stream.next()).value!;
      const first = (await stream.next()).value!; const second = (await stream.next()).value!;
      const clock = videoClock(init.data);
      assert.equal(boxes(first.data)[0].type, 'prft');
      assert.ok(decodeTime(second.data, clock.track) > decodeTime(first.data, clock.track));
      const file = join(root, 'clip.mp4'); await writeFile(file, Buffer.concat([init.data, first.data, second.data]));
      await execute('ffmpeg', ['-v', 'error', '-i', file, '-f', 'null', '-'], { timeout: 15000 });
      const probe = await execute('ffprobe', ['-v', 'error', '-select_streams', 'v', '-show_packets', '-of', 'json', file]);
      const packets = JSON.parse(probe.stdout).packets;
      assert.match(packets[0].flags, /K/); assert.ok(packets.length >= 100);
      recording.closeRecordingStream(1); await stream.return(undefined);
      await recording.close(); assert.equal(recording.ring.size, 0);
    });
    await t.test('local transport sends decryptable HEVC and Opus with prepared SSRCs', async () => {
      const receiver = await udp(); const audioReceiver = await udp();
      try {
        const parameters = { cryptoSuite: 0, masterKey: Buffer.alloc(16, 1), masterSalt: Buffer.alloc(14, 2) };
        const result = await local.prepareStream({ sessionIdentifier: 'local', addressVersion: 'ipv4',
          targetAddress: '127.0.0.1', sourceAddress: '127.0.0.1', controllerVideoPort: receiver.address().port,
          controllerAudioPort: audioReceiver.address().port, video: parameters, audio: parameters });
        const decrypt = (value: typeof result.video) => new SrtpSession({ profile: 1, keys: {
          localMasterKey: value.masterKey, localMasterSalt: value.masterSalt,
          remoteMasterKey: value.masterKey, remoteMasterSalt: value.masterSalt,
        } });
        const videoCipher = decrypt(result.video); const audioCipher = decrypt(result.audio);
        let videoCount = 0; let audioCount = 0;
        receiver.on('message', bytes => {
          if (bytes[1] >= 192 && bytes[1] <= 223) return;
          const packet = RtpPacket.deSerialize(videoCipher.decrypt(bytes));
          assert.equal(packet.header.ssrc, result.videoSSRC); assert.equal(packet.header.payloadType, 99); videoCount++;
        });
        audioReceiver.on('message', bytes => {
          if (bytes[1] >= 192 && bytes[1] <= 223) return;
          const packet = RtpPacket.deSerialize(audioCipher.decrypt(bytes));
          assert.equal(packet.header.ssrc, result.audioSSRC); assert.equal(packet.header.payloadType, 110); audioCount++;
        });
        await local.startStream({ sessionIdentifier: 'local', videoTier: 3, videoSSRC: 999, audioTier: 1, audioSSRC: 998 });
        await until(() => videoCount > 10 && audioCount > 10); await local.stopStream('local');
        assert.equal(local.sessions.size, 0);
      } finally { receiver.close(); audioReceiver.close(); }
    });
    await t.test('remote SDP carries SFrame-authenticated HEVC and Opus through DTLS-SRTP', async st => {
      const viewer = new RTCPeerConnection({ bundlePolicy: 'max-bundle', iceServers: [], iceUseIpv6: false,
        iceAdditionalHostAddresses: ['127.0.0.1'], iceInterfaceAddresses: { udp4: '127.0.0.1' },
        headerExtensions: { video: [new RTCRtpHeaderExtensionParameters({ id: 1, uri: STREAM_ID_URI })], audio: [] }, codecs: {
        video: [new RTCRtpCodecParameters({ mimeType: 'video/H265', clockRate: 90000, payloadType: 99,
          parameters: 'profile-id=1;tier-flag=0;level-id=153;tx-mode=SRST' })],
        audio: [new RTCRtpCodecParameters({ mimeType: 'audio/opus', clockRate: 48000, channels: 2, payloadType: 110 })],
      } });
      try {
        const offer = await remote.handleSolicitOffer({ sessionIdentifier: 'remote', options: { sframeEnabled: true } });
        assert.ok(offer.sframe); let packets = 0; let videoFrames = 0; let videoPackets = 0;
        const failures: string[] = [];
        viewer.onTrack.subscribe(track => {
          const depacketizer = new SFrameRtpDepacketizer();
          track.onReceiveRtp.subscribe(packet => {
            if (track.kind === 'video') videoPackets++;
            try {
            const sealed = depacketizer.push(packet); if (!sealed) return;
            const cryptor = new SFrameReceiver(new Map([[offer.sframe!.kid, offer.sframe!.key]]), packet.header.ssrc,
              track.kind === 'audio' ? SFrameCipherSuite.AES_256_CTR_HMAC_SHA512_32 : SFrameCipherSuite.AES_256_CTR_HMAC_SHA512_80);
            const clear = cryptor.unprotectFrame(sealed); assert.ok(clear.length > 0);
            if (track.kind === 'audio') packets++;
            else { assert.ok(clear.readUInt32BE(0) <= clear.length - 4); videoFrames++; }
            } catch (error) { if (failures.length < 5) failures.push(String(error)); }
          });
        });
        await viewer.setRemoteDescription({ type: 'offer', sdp: offer.sdpOffer });
        await viewer.setLocalDescription(await viewer.createAnswer());
        await remote.handleProvideAnswer({ sessionIdentifier: 'remote', sdpAnswer: viewer.localDescription!.sdp, candidates: [] });
        try { await until(() => packets > 5 && videoFrames > 5, 20000); }
        catch (error) { st.diagnostic(JSON.stringify({ packets, videoPackets, videoFrames, failures, state: viewer.connectionState, remote: remote.counters })); throw error; }
        assert.deepEqual(failures, []);
        await remote.handleEndSession('remote'); assert.equal(remote.sessions.size, 0);
      } finally { await viewer.close(); }
    });
    await t.test('privacy stops producers and rejects new transports', async () => {
      allowed = false;
      await Promise.all([local.close(), remote.close(), recording.close()]); await media.stop();
      assert.equal(media.children.size, 0);
      await assert.rejects(remote.handleSolicitOffer({ sessionIdentifier: 'off', options: { sframeEnabled: true } }), /unavailable/);
    });
  } finally {
    await Promise.all([recording.close(), local.close(), remote.close()]); await media.stop();
    await rm(root, { recursive: true, force: true });
  }
});
