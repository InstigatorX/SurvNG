import { performance } from 'node:perf_hooks';

export type AgentPhase = 'startup' | 'version' | 'sync' | 'sync body' | 'sync cleanup' | 'retry'
  | 'updates' | 'assignment cleanup' | 'feed cleanup' | 'snapshots' | 'shutdown';

export class AgentProgress {
  phase: AgentPhase = 'startup';
  deadline = performance.now() + 15000;
  enter(phase: AgentPhase, timeout = 30000): void {
    this.phase = phase;
    this.deadline = performance.now() + timeout;
  }
}

/** Own the daemon's lifetime, including waits backed only by unreferenced timers.
 * If a wait never completes, abort it and bound shutdown before systemd reaps
 * the service's remaining children and restarts it.
 */
export function superviseAgent(running: Promise<unknown>, shutdown: AbortController,
  progress = new AgentProgress(), options: { tickMs?: number; graceMs?: number } = {}): void {
  let requestedStop = false;
  let stopping = false;
  let grace: NodeJS.Timeout | undefined;
  const stop = (requested: boolean) => {
    if (stopping) return;
    stopping = true;
    requestedStop = requested;
    clearInterval(lifetime);
    if (!requested) {
      process.exitCode = 1;
      console.error(`SurvNG HomeKit encoder stalled during ${progress.phase}; requesting restart.`);
    }
    grace = setTimeout(() => {
      console.error(`SurvNG HomeKit encoder shutdown incomplete during ${progress.phase}.`);
      process.exit(1);
    }, options.graceMs ?? 5000);
    shutdown.abort();
  };
  // This interval deliberately stays referenced. AbortSignal.timeout() does
  // not keep Node alive, and pending promises alone do not own daemon lifetime.
  const lifetime = setInterval(() => {
    if (performance.now() >= progress.deadline) stop(false);
  }, options.tickMs ?? 1000);
  const signalStop = () => stop(true);
  process.once('SIGTERM', signalStop);
  process.once('SIGINT', signalStop);
  const finish = () => {
    clearInterval(lifetime); clearTimeout(grace);
    process.removeListener('SIGTERM', signalStop);
    process.removeListener('SIGINT', signalStop);
  };
  void running.then(finish, () => { process.exitCode = 1; finish(); });
  process.once('beforeExit', () => {
    if (!requestedStop && !process.exitCode) {
      console.error(`SurvNG HomeKit encoder exited unexpectedly during ${progress.phase}; requesting restart.`);
      process.exitCode = 1;
    }
  });
}
