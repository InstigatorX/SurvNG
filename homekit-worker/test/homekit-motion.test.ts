import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Accessory, Characteristic, decodeCameraZones, encodeCameraZones, SecureVideoController, StreamTierVideoCodec, uuid, ZoneApplicationMethod,
  type SecureVideoControllerOptions } from '@homebridge/hap-nodejs';
import { LabRecording } from '../src/fragments.js';
import { HomeKitMotionController, motionZonesBlockUnlocatedMotion } from '../src/homekit-motion.js';
import { SyntheticMedia, tiers } from '../src/media.js';
import { LocalStreams, RemoteStreams } from '../src/transports.js';

function options(): SecureVideoControllerOptions {
  const media = new SyntheticMedia('/unused-motion-fixture');
  const allowed = () => true;
  return {
    sensor: { uuid: uuid.generate('motion-sensor'), width: 1920, height: 1080 },
    video: { codec: StreamTierVideoCodec.H265, tiers },
    rtp: { delegate: new LocalStreams(media, allowed) },
    webrtc: { delegate: new RemoteStreams(media, allowed, '127.0.0.1') },
    recording: {
      delegate: new LabRecording(media, allowed),
      options: {
        prebufferLength: 4000,
        mediaContainerConfiguration: { type: 0, fragmentLength: 4000 },
        video: { type: 0, parameters: { profiles: [1], levels: [2] }, resolutions: [[1920, 1080, 30]] },
        audio: { codecs: [{ type: 0, audioChannels: 1, bitrateMode: 0, samplerate: 3 }] },
      },
    },
  };
}

test('fresh internal motion is enabled without changing the controller identity', () => {
  const controller = new HomeKitMotionController(options());
  const accessory = new Accessory('Camera fixture', uuid.generate('fresh-camera'));
  accessory.configureController(controller);
  assert.equal(controller.controllerId(), new SecureVideoController(options()).controllerId());
  assert.equal(controller.motionService?.getCharacteristic(Characteristic.MotionEnabled).value, true);
});

test('a paired camera can restore its explicitly disabled motion state', () => {
  const controller = new HomeKitMotionController(options());
  const accessory = new Accessory('Camera fixture', uuid.generate('persisted-camera'));
  accessory.configureController(controller);
  controller.motionService!.updateCharacteristic(Characteristic.MotionEnabled, false);
  const saved = controller.serialize()!;

  const restored = new HomeKitMotionController(options());
  Accessory.deserialize(Accessory.serialize(accessory)).configureController(restored);
  restored.deserialize(saved);
  assert.equal(restored.motionService?.getCharacteristic(Characteristic.MotionEnabled).value, false);
});

test('only active, populated activity zones block unlocated motion', () => {
  const emptyZones = decodeCameraZones(encodeCameraZones({ version: 1, zones: [] }));
  const configuredZones = { version: 1, zones: [{ method: ZoneApplicationMethod.NORMAL, polygons: [{ identifier: uuid.generate('zone'),
    vertices: [{ x: 0, y: 0 }, { x: 50, y: 0 }, { x: 50, y: 50 }] }] }] };
  assert.equal(motionZonesBlockUnlocatedMotion({ active: true }), false);
  assert.equal(motionZonesBlockUnlocatedMotion({ active: true, zones: emptyZones }), false);
  assert.equal(motionZonesBlockUnlocatedMotion({ active: false, zones: configuredZones }), false);
  assert.equal(motionZonesBlockUnlocatedMotion({ active: true, zones: configuredZones }), true);
});
