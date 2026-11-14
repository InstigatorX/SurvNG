import { Characteristic, Service } from '@homebridge/hap-nodejs';
import { RtcpRrPacket, RtcpSrPacket, RtcpPayloadSpecificFeedback, PictureLossIndication, type RtcpPacket } from 'werift';

/** Bounded receiver evidence only; never retain packets, identifiers, or key material. */
export class ReceiverFeedback {
  readonly stats = { reports: 0, maxPacketsLost: 0, maxFractionLost: 0, maxJitterMs: 0, keyframeRequests: 0,
    packetsLost: 0, fractionLost: 0, jitterMs: 0 };
  constructor(private clockRate: number) {}
  observe(packet: RtcpPacket, senderSSRC: number): void {
    if (packet instanceof RtcpRrPacket || packet instanceof RtcpSrPacket) {
      for (const report of packet.reports) {
        if (report.ssrc !== senderSSRC) continue;
        this.stats.reports++;
        // RTCP cumulative loss is signed 24-bit; werift exposes its wire bits unsigned.
        const lost = report.packetsLost > 0x7fffff ? report.packetsLost - 0x1000000 : report.packetsLost;
        this.stats.packetsLost = lost;
        this.stats.fractionLost = report.fractionLost / 256;
        this.stats.jitterMs = report.jitter / this.clockRate * 1000;
        this.stats.maxPacketsLost = Math.max(this.stats.maxPacketsLost, lost);
        this.stats.maxFractionLost = Math.max(this.stats.maxFractionLost, report.fractionLost / 256);
        this.stats.maxJitterMs = Math.max(this.stats.maxJitterMs, report.jitter / this.clockRate * 1000);
      }
    } else if (packet instanceof RtcpPayloadSpecificFeedback) {
      const feedback = packet.feedback;
      if (feedback instanceof PictureLossIndication && feedback.mediaSsrc === senderSSRC
        || feedback.count === 4 && 'fir' in feedback && feedback.fir.some(entry => entry.ssrc === senderSSRC)) {
        this.stats.keyframeRequests++;
      }
    }
  }
}

// Read cached, allowlisted state only. Never invoke handlers or expose transport TLVs.
export function serviceStatus(service: Service | undefined) {
  if (!service) return undefined;
  const flags = [Characteristic.HomeKitCameraActive, Characteristic.StreamingEnabled,
    Characteristic.StatusActive, Characteristic.ManuallyDisabled, Characteristic.ThirdPartyCameraActive,
    Characteristic.EventSnapshotsActive, Characteristic.PeriodicSnapshotsActive, Characteristic.Version,
    Characteristic.Mute, Characteristic.Volume];
  return {
    iid: service.iid,
    linkedIids: service.linkedServices.map(linked => linked.iid),
    errors: service.characteristics.filter(characteristic => characteristic.statusCode !== 0)
      .map(characteristic => ({ iid: characteristic.iid, name: characteristic.displayName, status: characteristic.statusCode })),
    flags: Object.fromEntries(flags.flatMap(type => {
      const characteristic = service.characteristics.find(candidate => candidate.UUID === type.UUID);
      if (!characteristic) return [];
      const value = characteristic.value;
      return [[characteristic.displayName, { iid: characteristic.iid, status: characteristic.statusCode,
        value: typeof value === 'boolean' || typeof value === 'number'
          || (type === Characteristic.Version && typeof value === 'string') ? value : null }]];
    })),
  };
}
