import { randomBytes, randomInt, randomUUID } from 'node:crypto';
import { mkdir, lstat, open, readFile, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { createServer, type Server, type Socket } from 'node:net';

export interface Identity { id: string; username: string; pincode: string }
export async function privateDirectory(path: string): Promise<void> {
  await mkdir(path, { recursive: true, mode: 0o700 });
  const info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink() || info.uid !== process.getuid?.() || (info.mode & 0o077)) {
    throw new Error('state directory must be owned by this user with mode 0700');
  }
}
export async function identity(path: string): Promise<Identity> {
  const filename = join(path, 'identity.json');
  try {
    const info = await lstat(filename);
    if (!info.isFile() || info.isSymbolicLink() || info.uid !== process.getuid?.() || (info.mode & 0o077)) throw new Error('unsafe identity permissions');
    const value = JSON.parse(await readFile(filename, 'utf8')) as Identity;
    if (!/^[0-9a-f-]{36}$/i.test(value.id) || !/^([0-9A-F]{2}:){5}[0-9A-F]{2}$/.test(value.username) || !/^\d{3}-\d{2}-\d{3}$/.test(value.pincode)) {
      throw new Error('invalid saved identity');
    }
    return value;
  } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error; }
  let pin: string;
  do { pin = randomInt(10000000, 99999999).toString(); } while (/^(\d)\1+$/.test(pin) || pin === '12345678' || pin === '87654321');
  const value: Identity = { id: randomUUID(), username: randomBytes(6).toString('hex').toUpperCase().match(/../g)!.join(':'),
    pincode: `${pin.slice(0, 3)}-${pin.slice(3, 5)}-${pin.slice(5)}` };
  const temporary = join(path, `identity-${randomUUID()}.tmp`);
  const file = await open(temporary, 'wx', 0o600);
  try { await file.writeFile(JSON.stringify(value)); await file.sync(); } finally { await file.close(); }
  await rename(temporary, filename);
  return value;
}

/** A private lab socket, deliberately separate from production observability. */
export async function controlServer(path: string, handler: (method: string, params: unknown) => unknown | Promise<unknown>): Promise<{ server: Server; close(): Promise<void> }> {
  const clients = new Set<Socket>();
  const server = createServer(socket => {
    clients.add(socket); socket.on('close', () => clients.delete(socket));
    socket.on('error', () => socket.destroy()); socket.setTimeout(5000, () => socket.destroy());
    let input: Buffer = Buffer.alloc(0); let received = false;
    socket.on('data', (chunk: Buffer) => {
      if (received) { socket.destroy(); return; }
      input = Buffer.concat([input, chunk]);
      if (input.length > 16_384) { socket.destroy(); return; }
      const end = input.indexOf(10); if (end < 0) return;
      received = true;
      let request: { version: number; id: string; method: string; params?: unknown };
      try {
        request = JSON.parse(input.subarray(0, end).toString('utf8'));
        if (request.version !== 1 || typeof request.id !== 'string' || request.id.length > 80 || typeof request.method !== 'string') throw new Error();
      } catch { socket.end(JSON.stringify({ version: 1, error: 'invalid_request' }) + '\n'); return; }
      Promise.resolve().then(() => handler(request.method, request.params)).then(
        result => socket.end(JSON.stringify({ version: 1, id: request.id, result }) + '\n'),
        () => socket.end(JSON.stringify({ version: 1, id: request.id, error: 'request_failed' }) + '\n'),
      );
    });
  });
  // Never unlink a pre-existing socket: it may belong to a running instance.
  await new Promise<void>((resolve, reject) => { server.once('error', reject); server.listen(path, resolve); });
  return { server, close: () => new Promise<void>((resolve, reject) => {
    for (const client of clients) client.destroy();
    server.close(error => error ? reject(error) : resolve());
  }) };
}
