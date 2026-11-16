import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';

const lifecycle = new URL('../src/agent-lifecycle.js', import.meta.url).href;

test('a stalled daemon has a bounded failed shutdown', async () => {
  const child = spawn(process.execPath, ['--input-type=module', '-e', `
    import { AgentProgress, superviseAgent } from ${JSON.stringify(lifecycle)};
    const progress = new AgentProgress(); progress.enter('sync', 40);
    superviseAgent(new Promise(() => {}), new AbortController(), progress, { tickMs: 10, graceMs: 40 });
  `], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  child.stderr.on('data', chunk => stderr += chunk.toString());
  const [code] = await once(child, 'close');
  assert.equal(code, 1);
  assert.match(stderr, /stalled during sync/);
  assert.match(stderr, /shutdown incomplete/);
});

test('daemon lifetime keeps an unreferenced AbortSignal timeout alive', async () => {
  const child = spawn(process.execPath, ['--input-type=module', '-e', `
    import { superviseAgent } from ${JSON.stringify(lifecycle)};
    const signal = AbortSignal.timeout(80);
    const running = new Promise(resolve => signal.addEventListener('abort', () => { console.log('completed'); resolve(); }));
    superviseAgent(running, new AbortController());
  `], { stdio: ['ignore', 'pipe', 'pipe'] });
  let stdout = '';
  child.stdout.on('data', chunk => stdout += chunk.toString());
  child.stderr.resume();
  const [code] = await once(child, 'close');
  assert.match(stdout, /completed/, 'the pending operation must finish before exit');
  assert.equal(code, 1, 'an unsolicited end of the daemon still requests a restart');
});

test('a requested service stop still drains and exits successfully', async () => {
  const child = spawn(process.execPath, ['--input-type=module', '-e', `
    import { superviseAgent } from ${JSON.stringify(lifecycle)};
    const shutdown = new AbortController();
    const timer = setInterval(() => {}, 1000);
    const running = new Promise(resolve => shutdown.signal.addEventListener('abort', () => {
      clearInterval(timer); resolve();
    }));
    superviseAgent(running, shutdown);
    console.log('ready');
  `], { stdio: ['ignore', 'pipe', 'pipe'] });
  await once(child.stdout, 'data');
  child.kill('SIGTERM');
  assert.equal((await once(child, 'close'))[0], 0);
});
