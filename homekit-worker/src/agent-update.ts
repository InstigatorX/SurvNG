import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { existsSync } from 'node:fs';
import { mkdtemp, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';

async function listing(file: string, verbose: boolean): Promise<string[]> {
  const child = spawn('tar', [verbose ? '-tvzf' : '-tzf', file, '--numeric-owner'], {
    stdio: ['ignore', 'pipe', 'ignore'], timeout: 30000, killSignal: 'SIGKILL' });
  const chunks: Buffer[] = [];
  let size = 0;
  child.stdout.on('data', chunk => {
    size += chunk.length;
    if (size > 8 * 1024 * 1024) child.kill('SIGKILL'); else chunks.push(chunk);
  });
  const [code] = await once(child, 'close');
  if (code !== 0 || size > 8 * 1024 * 1024) throw new Error('update rejected');
  return Buffer.concat(chunks).toString('utf8').split('\n').filter(Boolean);
}

export async function validateArchive(file: string, allow: (name: string) => boolean): Promise<string[]> {
  const names = await listing(file, false);
  if (!names.length || names.some(name => name.startsWith('/') || name.includes('\\') || name.split('/').includes('..') || !allow(name))) {
    throw new Error('update rejected');
  }
  // Names alone do not confine links or device entries to the private staging directory.
  const types = await listing(file, true);
  if (types.length !== names.length || types.some(line => line[0] !== '-' && line[0] !== 'd')) throw new Error('update rejected');
  return names;
}

/** All components are staged first; a failed rename restores every earlier component. */
export async function replaceUpdate(home: string, entries: { name: string; source: string }[]): Promise<void> {
  if (entries.some(entry => !['dist', 'node_modules', 'package.json', 'bin', 'lib', '.releases.json'].includes(entry.name))
      || new Set(entries.map(entry => entry.name)).size !== entries.length) throw new Error('invalid update');
  const backup = await mkdtemp(join(home, '.rollback-'));
  const moved: { destination: string; backup: string; previous: boolean; installed: boolean }[] = [];
  try {
    for (const entry of entries) {
      const destination = join(home, entry.name);
      const saved = join(backup, entry.name);
      const previous = existsSync(destination);
      if (previous) await rename(destination, saved);
      const state = { destination, backup: saved, previous, installed: false };
      moved.push(state);
      await rename(entry.source, destination);
      state.installed = true;
    }
  } catch (error) {
    for (const entry of moved.reverse()) {
      if (entry.installed) await rm(entry.destination, { recursive: true, force: true });
      if (entry.previous) await rename(entry.backup, entry.destination);
    }
    await rm(backup, { recursive: true, force: true });
    throw error;
  }
  // At this point the release, including its stamp, has committed. Cleanup must
  // not make the caller resume the old process against the new installation.
  try { await rm(backup, { recursive: true, force: true }); }
  catch { console.error('SurvNG HomeKit update backup cleanup failed.'); }
}
