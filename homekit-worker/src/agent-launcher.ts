// Installed as dist/src/agent.js in the service working directory. Keep this
// boot/recovery path independent of the candidates and their dependencies.
import { fork, spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { closeSync, fsyncSync, openSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { readdir, rm } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const generationStore = '.encoder-generations';
const bootFile = '.encoder-boot.json';
const generationName = /^gen-[A-Za-z0-9]{6}$/;

// The fixed legacy tree remains available until migration's final rename.
// Node would otherwise walk up to its node_modules for missing candidate files.
const generationResolver = `
  import { realpathSync } from 'node:fs';
  import { registerHooks } from 'node:module';
  import { pathToFileURL } from 'node:url';
  const generation = pathToFileURL(realpathSync(process.cwd()) + '/').href;
  registerHooks({
    resolve(specifier, context, nextResolve) {
      const result = nextResolve(specifier, context);
      if (result.url.startsWith('file:') && !result.url.startsWith(generation)) {
        throw new Error('Encoder module resolved outside its generation.');
      }
      return result;
    }
  });
`;
export const generationArgs = ['--import', 'data:text/javascript,' + encodeURIComponent(generationResolver)];

type Releases = Partial<Record<'agent' | 'ffmpeg' | 'runtime' | 'syncRequest', string>>;
export type BootRecord = {
  version: 1;
  good: string;
  trial?: { id: string; attempted: boolean };
  rejected?: Releases;
};

export function syncDirectory(directory: string): void {
  const descriptor = openSync(directory, 'r');
  try { fsyncSync(descriptor); } finally { closeSync(descriptor); }
}

export function readBoot(root: string): BootRecord {
  const state = JSON.parse(readFileSync(join(root, bootFile), 'utf8')) as BootRecord;
  if (state?.version !== 1 || typeof state.good !== 'string' || !generationName.test(state.good)
      || (state.trial !== undefined && (!state.trial || typeof state.trial.id !== 'string'
        || !generationName.test(state.trial.id) || state.trial.id === state.good || typeof state.trial.attempted !== 'boolean'))) {
    throw new Error('invalid encoder boot record');
  }
  return state;
}

export function writeBoot(root: string, record: BootRecord): void {
  const temp = join(root, bootFile + '.' + randomUUID());
  const descriptor = openSync(temp, 'wx', 0o600);
  try {
    try {
      writeFileSync(descriptor, JSON.stringify(record));
      fsyncSync(descriptor);
    } finally { closeSync(descriptor); }
    renameSync(temp, join(root, bootFile));
    syncDirectory(root);
  } finally { rmSync(temp, { force: true }); }
}

function rejectTrial(root: string, record: BootRecord): BootRecord {
  const path = join(root, generationStore, record.trial!.id, '.releases.json');
  let rejected: Releases = {};
  try {
    const releases = JSON.parse(readFileSync(path, 'utf8'));
    for (const kind of ['agent', 'ffmpeg', 'runtime', 'syncRequest'] as const) {
      if (typeof releases[kind] === 'string') rejected[kind] = releases[kind];
    }
  } catch { console.error('Encoder could not read the rejected release stamp.'); }
  const recovered: BootRecord = { version: 1, good: record.good, rejected };
  writeBoot(root, recovered);
  console.error('Encoder startup trial failed. Restoring the previous generation.');
  return recovered;
}

export async function pruneEncoder(root: string, record?: BootRecord): Promise<void> {
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (entry.isDirectory() && /^\.(?:update-(?:stamp-)?|encoder-migrate-)[A-Za-z0-9]{6}$/.test(entry.name)) {
      await rm(join(root, entry.name), { recursive: true, force: true });
    }
  }
  const store = join(root, generationStore);
  const names = await readdir(store).catch(error => {
    if (error.code === 'ENOENT' && !record) return [];
    throw error;
  });
  for (const name of names) {
    if (generationName.test(name) && name !== record?.good && name !== record?.trial?.id) {
      await rm(join(store, name), { recursive: true, force: true });
    }
  }
}

// A stopped service is the exclusive owner during local migration. At runtime
// only the child queues a trial, and only the launcher attempts or confirms it.
export async function launchEncoder(root: string, readinessMs = 30000): Promise<number> {
  root = resolve(root);
  let record = readBoot(root);
  if (record.trial?.attempted) record = rejectTrial(root, record);
  try { await pruneEncoder(root, record); }
  catch { console.error('Encoder generation cleanup failed.'); }
  const trial = record.trial;
  if (trial) writeBoot(root, { ...record, trial: { id: trial.id, attempted: true } });
  const generation = join(root, generationStore, trial?.id ?? record.good);
  // Legacy units name root/lib in LD_LIBRARY_PATH. Keep all bundled lookups in
  // the selected generation, including the dynamic loader's own search path.
  const library = join(generation, 'lib');
  const env = { ...process.env, SURVNG_ENCODER_HOME: root, SURVNG_ENCODER_REJECTED: JSON.stringify(record.rejected ?? {}),
    SURVNG_FFMPEG: join(generation, 'bin', 'ffmpeg'), LIBVA_DRIVERS_PATH: join(library, 'dri'),
    LD_LIBRARY_PATH: [library, ...(process.env.LD_LIBRARY_PATH || '').split(':').filter(p => p && p !== join(root, 'lib'))].join(':') };
  const child = fork(join(generation, 'dist/src/agent.js'), [], { cwd: generation, env, execArgv: generationArgs, stdio: ['inherit', 'inherit', 'inherit', 'ipc'] });
  let ready = false;
  let stopping = false;
  let expired = false;
  const forward = (signal: NodeJS.Signals) => { stopping = true; child.kill(signal); };
  const terminate = () => forward('SIGTERM');
  const interrupt = () => forward('SIGINT');
  process.on('SIGTERM', terminate);
  process.on('SIGINT', interrupt);
  const timeout = trial ? setTimeout(() => { expired = true; child.kill('SIGKILL'); }, readinessMs) : undefined;
  try {
    return await new Promise<number>((done, fail) => {
      child.on('message', message => {
        if (ready || stopping || expired || message !== 'ready') return;
        try {
          if (trial) writeBoot(root, { version: 1, good: trial.id });
          ready = true;
          clearTimeout(timeout);
          if (child.connected) child.send('confirmed', () => {});
        } catch (error) {
          expired = true;
          child.kill('SIGKILL');
          fail(error);
        }
      });
      child.once('error', fail);
      child.once('close', code => {
        try {
          if (trial && !ready && !stopping) rejectTrial(root, record);
          done(stopping ? 0 : (trial && !ready ? 1 : code ?? 1));
        } catch (error) { fail(error); }
      });
    });
  } finally {
    clearTimeout(timeout);
    process.off('SIGTERM', terminate);
    process.off('SIGINT', interrupt);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === '--migrate' && process.argv.length === 5) {
    const bundle = resolve(process.argv[4]);
    const migration = spawnSync(process.execPath, [join(bundle, 'dist/src/agent-generations.js'), '--migrate', resolve(process.argv[3]), bundle],
      { stdio: 'inherit' });
    process.exitCode = migration.status ?? 1;
  } else if (process.argv.length === 2) {
    launchEncoder(process.cwd()).then(code => { process.exitCode = code; }).catch(() => {
      console.error('Encoder launcher could not start the selected generation.');
      process.exitCode = 1;
    });
  } else { process.exitCode = 1; }
}
