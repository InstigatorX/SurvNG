import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Characteristic, HAPStatus, Service } from '@homebridge/hap-nodejs';
import { RtcpRrPacket, RtcpReceiverInfo, RtcpPayloadSpecificFeedback, PictureLossIndication } from 'werift';
import { ReceiverFeedback, serviceStatus } from '../src/diagnostics.js';

test('service diagnostics separate mode flags and exclude transport values', () => {
  const local = new Service.CameraMultiTierRTPStreamManagement('', '');
  const remote = new Service.CameraWebRTCStreamManagement('', '');
  local.iid = 10; remote.iid = 20; local.addLinkedService(remote);
  local.setCharacteristic(Characteristic.StreamingEnabled, true);
  remote.setCharacteristic(Characteristic.StreamingEnabled, false);
  const secret = Buffer.from('private transport material').toString('base64');
  local.setCharacteristic(Characteristic.SetupEndpoints, secret);
  local.getCharacteristic(Characteristic.StatusActive).statusCode = HAPStatus.SERVICE_COMMUNICATION_FAILURE;
  const result = serviceStatus(local)!;
  assert.deepEqual(result.linkedIids, [20]);
  assert.equal(result.flags['Streaming Enabled'].value, true);
  assert.equal(result.flags['Status Active'].status, HAPStatus.SERVICE_COMMUNICATION_FAILURE);
  assert.deepEqual(result.errors, [{ iid: null, name: 'Status Active', status: HAPStatus.SERVICE_COMMUNICATION_FAILURE }]);
  assert.equal(serviceStatus(remote)!.flags['Streaming Enabled'].value, false);
  assert.equal(Object.hasOwn(result.flags, 'Setup Endpoints'), false);
  assert.equal(JSON.stringify(result).includes('private'), false);
  assert.equal(JSON.stringify(result).includes(secret), false);
  assert.equal(serviceStatus(undefined), undefined);
});

test('audio diagnostics expose cached mute and volume without invoking a read handler', () => {
  const microphone = new Service.Microphone('', '');
  microphone.setCharacteristic(Characteristic.Mute, false);
  microphone.setCharacteristic(Characteristic.Volume, 100);
  let reads = 0;
  microphone.getCharacteristic(Characteristic.Mute).onGet(() => { reads++; throw new Error('must not invoke hardware'); });
  const result = serviceStatus(microphone)!;
  assert.equal(result.flags.Mute.value, false);
  assert.equal(result.flags.Volume.value, 100);
  assert.equal(reads, 0);
});

test('receiver diagnostics retain only bounded loss, jitter, and keyframe counters', () => {
  const feedback = new ReceiverFeedback(90000);
  feedback.observe(new RtcpRrPacket({ reports: [
    new RtcpReceiverInfo({ ssrc: 9, packetsLost: 999, fractionLost: 255, jitter: 90000 }),
    new RtcpReceiverInfo({ ssrc: 1, packetsLost: 0xffffff, fractionLost: 0, jitter: 0 }),
    new RtcpReceiverInfo({ ssrc: 1, packetsLost: 3, fractionLost: 64, jitter: 900 }),
  ] }), 1);
  feedback.observe(new RtcpPayloadSpecificFeedback({ feedback: new PictureLossIndication({ mediaSsrc: 1 }) }), 1);
  feedback.observe(new RtcpPayloadSpecificFeedback({ feedback: new PictureLossIndication({ mediaSsrc: 9 }) }), 1);
  assert.deepEqual(feedback.stats, { reports: 2, maxPacketsLost: 3, maxFractionLost: 0.25,
    maxJitterMs: 10, keyframeRequests: 1 });
  assert.equal(Object.hasOwn(feedback.stats, 'ssrc'), false);
});
