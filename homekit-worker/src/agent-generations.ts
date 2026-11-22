import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { once } from 'node:events';
import { constants } from 'node:fs';
import { chmod, copyFile, cp, lstat, mkdir, mkdtemp, open, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { generationArgs, generationStore, pruneEncoder, readBoot, syncDirectory, writeBoot, type BootRecord } from './agent-launcher.js';

const components = ['dist', 'node_modules', 'package.json', 'bin', 'lib', '.releases.json'];
type Entry = { name: string; source: string };

function generationId(root: string, candidate: string): string {
  const id = basename(candidate);
  if (!/^gen-[A-Za-z0-9]{6}$/.test(id) || resolve(candidate) !== join(resolve(root), generationStore, id)) {
    throw new Error('invalid encoder generation');
  }
  return id;
}

export async function stageGeneration(installRoot: string, current: string, entries: Entry[]): Promise<string> {
  const paths = new Map(entries.map(entry => [entry.name, entry.source]));
  if (paths.size !== entries.length || [...paths.keys()].some(name => !components.includes(name))) {
    throw new Error('invalid encoder update');
  }
  const store = join(resolve(installRoot), generationStore);
  await mkdir(store, { recursive: true, mode: 0o700 });
  const candidate = await mkdtemp(join(store, 'gen-'));
  try {
    for (const name of components) {
      const source = paths.get(name) ?? join(current, name);
      // Historical installs can predate a release stamp.
      if (name === '.releases.json' && !paths.has(name)) {
        try { await lstat(source); }
        catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') continue; throw error; }
      }
      await cp(source, join(candidate, name), {
        recursive: true, mode: constants.COPYFILE_FICLONE, verbatimSymlinks: true,
      });
    }
    // Linux syncfs flushes the complete staged tree, including directory
    // entries and binaries, before its name can reach a durable boot record.
    const sync = spawn('sync', ['-f', candidate], { stdio: 'ignore' });
    const [code] = await once(sync, 'close');
    if (code !== 0) throw new Error('encoder generation sync failed');
    return candidate;
  } catch (error) {
    await rm(candidate, { recursive: true, force: true });
    throw error;
  }
}

export async function activateGeneration(root: string, candidate: string): Promise<void> {
  const id = generationId(root, candidate);
  const state = readBoot(root);
  if (state.trial || state.good === id) throw new Error('encoder generation already selected');
  try {
    writeBoot(root, { ...state, trial: { id, attempted: false } });
  } catch (error) {
    // The rename can succeed before the parent-directory fsync fails. Retire
    // the running agent if the visible record already names this candidate.
    if (readBoot(root).trial?.id !== id) throw error;
    console.error('Encoder trial selected, but boot record durability could not be confirmed.');
  }
}

export async function discardGeneration(root: string, candidate: string): Promise<void> {
  const id = generationId(root, candidate);
  const state = readBoot(root);
  if (id !== state.good && id !== state.trial?.id) await rm(candidate, { recursive: true, force: true });
}

export async function preflightGeneration(candidate: string, signal?: AbortSignal): Promise<void> {
  const library = join(candidate, 'lib');
  const child = spawn(process.execPath, [...generationArgs, join(candidate, 'dist/src/agent.js'), '--preflight'], {
    cwd: candidate, env: { PATH: process.env.PATH, LD_LIBRARY_PATH: library, LIBVA_DRIVERS_PATH: join(library, 'dri'),
      SURVNG_FFMPEG: join(candidate, 'bin/ffmpeg') },
    stdio: 'ignore', timeout: 30000, killSignal: 'SIGKILL', signal,
  });
  const [code] = await once(child, 'close');
  if (code !== 0) throw new Error('encoder generation preflight failed');
}

async function installLauncher(root: string, bundle: string): Promise<void> {
  const destination = join(root, 'dist/src/agent.js');
  const temporary = destination + '.' + randomUUID();
  try {
    await copyFile(join(bundle, 'dist/src/agent-launcher.js'), temporary, constants.COPYFILE_EXCL);
    await chmod(temporary, (await lstat(destination)).mode & 0o777);
    const handle = await open(temporary, 'r');
    try { await handle.sync(); } finally { await handle.close(); }
    await rename(temporary, destination);
    syncDirectory(dirname(destination));
  } finally { await rm(temporary, { force: true }); }
}

async function localReleases(root: string): Promise<Record<string, string>> {
  try { return JSON.parse(await readFile(join(root, '.releases.json'), 'utf8')); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return {}; throw error; }
}

/** The operator stops the unit before invoking this from a trusted extracted bundle. */
export async function migrateGeneration(root: string, bundle: string): Promise<void> {
  root = resolve(root);
  bundle = resolve(bundle);
  let state: BootRecord | undefined;
  try { state = readBoot(root); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  const prior = root === bundle;
  await pruneEncoder(root, state);
  if (!state) {
    // Capture the original entry and tools before replacing the fixed systemd
    // target. A power cut before the final rename still boots the original.
    const baseline = await stageGeneration(root, root, []);
    if (prior) {
      try { await preflightGeneration(baseline); }
      catch (error) { await rm(baseline, { recursive: true, force: true }); throw error; }
    }
    state = { version: 1, good: generationId(root, baseline) };
    writeBoot(root, state);
  }
  if (!prior) {
    const agent = (await localReleases(bundle)).agent;
    if (typeof agent !== 'string' || !/^[0-9a-f]{64}$/.test(agent)) throw new Error('missing trusted agent release identity');
    const selected = join(root, generationStore, state.trial?.id ?? state.good);
    if ((await localReleases(selected)).agent !== agent || state.trial?.attempted) {
      const good = join(root, generationStore, state.good);
      if (state.trial) {
        const replaced = join(root, generationStore, state.trial.id);
        state = { ...state, trial: undefined };
        writeBoot(root, state);
        await discardGeneration(root, replaced);
      }
      const entries = ['dist', 'node_modules', 'package.json'].map(name => ({ name, source: join(bundle, name) }));
      const scratch = await mkdtemp(join(root, '.encoder-migrate-'));
      let candidate: string | undefined;
      try {
        const source = join(scratch, '.releases.json');
        const { ffmpeg, runtime } = await localReleases(good);
        await writeFile(source, JSON.stringify({ agent, ffmpeg, runtime }), { mode: 0o600 });
        candidate = await stageGeneration(root, good, [...entries, { name: '.releases.json', source }]);
        await preflightGeneration(candidate);
        writeBoot(root, { ...state, trial: { id: generationId(root, candidate), attempted: false } });
      } finally {
        await rm(scratch, { recursive: true, force: true });
        if (candidate) await discardGeneration(root, candidate);
      }
    }
  }
  await installLauncher(root, bundle);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url) && process.argv[2] === '--migrate' && process.argv.length === 5) {
  migrateGeneration(process.argv[3], process.argv[4]).catch(() => {
    console.error('Encoder migration failed. Verify the local release, tools and available disk space.');
    process.exitCode = 1;
  });
}
