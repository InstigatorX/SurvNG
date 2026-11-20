import { Characteristic, SecureVideoController } from '@homebridge/hap-nodejs';

export class HomeKitMotionController extends SecureVideoController {
  override constructServices() {
    const services = super.constructServices();
    // The pinned controller skips the default for a new internal sensor.
    // Persisted controller data can still restore an explicit disabled value.
    services.motionSensor?.updateCharacteristic(Characteristic.MotionEnabled, true);
    return services;
  }
}

export function motionZonesBlockUnlocatedMotion(motionZones: SecureVideoController['motionZones']): boolean {
  return motionZones.active && (motionZones.zones?.zones.length ?? 0) > 0;
}
