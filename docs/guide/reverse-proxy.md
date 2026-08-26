# Reverse proxy and internet access

SurvNG can sit behind nginx (or another reverse proxy) and can be reached from the internet **if you turn sign-in on** and let SurvNG trust only your proxy.

Do this in order. Skipping sign-in leaves the console open to anyone who can reach the port.

## 1. Turn on sign-in first

On the LAN, before you open a public port:

1. Open **Admin → Access**.
2. Create an **Admin** user (or enable sign-in and create one when SurvNG asks).
3. Enable **Require sign-in for the browser console**.
4. Set **Session length** if you want something other than 14 days.
5. **Save settings**.

Until sign-in is required, SurvNG is a LAN console: anyone who can load the site can change cameras and settings.

### First administrator from the internet

If you enable sign-in with **no users** and then open the site from a public address, SurvNG will ask for a **setup token**. That token is not shown in the browser.

On the server:

```bash
sudo cat /path/to/survng/storage/bootstrap.token
```

The path is `{storage_dir}/bootstrap.token` (often `survng/storage/bootstrap.token`, or `docker-data/storage/bootstrap.token` in Docker). SurvNG also writes a log line when it creates the file. Paste the token into the create-administrator screen.

Requests from private addresses (`192.168.x.x`, `10.x.x.x`, localhost) do not need the token. After the first admin exists, SurvNG deletes the file.

## 2. Tell SurvNG which proxy to trust

nginx (or Caddy, Traefik, Docker) talks to SurvNG on HTTP. It must send:

- `X-Forwarded-Proto: https` so cookies are marked **Secure** and SurvNG sends **HSTS**
- `X-Forwarded-For` with the real client IP (login lockout and the setup-token check)

SurvNG **ignores those headers** unless the connecting peer is in **Trusted reverse proxies** on **Admin → Access**.

Default:

```text
127.0.0.1
::1
```

That is correct when nginx runs **on the same computer** as SurvNG.

| How you run nginx | What to put in Trusted reverse proxies |
| --- | --- |
| nginx on the SurvNG host, `proxy_pass http://127.0.0.1:8088` | `127.0.0.1` and `::1` (default) |
| Docker Compose, nginx container → SurvNG container | the Docker network, often `172.16.0.0/12` plus `127.0.0.1` |
| Proxy on another machine | that proxy’s IP or CIDR only |

Do **not** set `*` unless SurvNG listens only on localhost and nothing else can reach port 8088. `*` means any client can fake HTTPS and client IPs.

Save settings after you change the list. SurvNG applies it immediately (no restart).

## 3. nginx example (HTTPS in front, SurvNG on localhost)

This assumes:

- Public site: `https://survng.example.com/survng/`
- SurvNG: `http://127.0.0.1:8088` with `base_path` `/survng` (the default)
- Sign-in is already on

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 80;
    listen [::]:80;
    server_name survng.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name survng.example.com;

    ssl_certificate     /etc/letsencrypt/live/survng.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/survng.example.com/privkey.pem;

    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    location = /survng {
        return 301 /survng/;
    }

    location /survng/ {
        proxy_pass http://127.0.0.1:8088;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_buffering off;
        proxy_read_timeout 3600s;
        client_max_body_size 4m;
    }
}
```

Notes:

- **No trailing slash** on `proxy_pass http://127.0.0.1:8088`. nginx must forward `/survng/` to SurvNG. SurvNG strips `base_path` itself.
- `Upgrade` / `Connection` are required for live camera WebSockets.
- `proxy_read_timeout 3600s` and `proxy_buffering off` keep live video from stalling.
- HTTP on port 80 should **redirect to HTTPS**. SurvNG cannot do that redirect when it only sees localhost HTTP from nginx.
- You can also set HSTS in nginx (shown above). SurvNG adds HSTS on responses it believes are HTTPS, which requires a trusted `X-Forwarded-Proto`.

If you serve SurvNG at the site root instead of `/survng/`, set **Web Base Path** to empty in Admin → General, Save, then use `location /` with the same `proxy_pass` and headers.

## 4. Built-in HTTPS (no nginx)

On a home LAN you can skip nginx:

1. **Admin → Access → HTTPS**: hostname, generate or upload a certificate, enable HTTPS, restart.
2. Start SurvNG with `python -m survng.app` (or the systemd/Docker command), not plain `uvicorn ...:app`.
3. Open `https://your-host:port/survng/`.

Browsers warn on a **self-signed** certificate until you trust it. For the public internet, prefer nginx or Caddy with Let’s Encrypt, and keep SurvNG on localhost.

## 5. Firewall

- Allow **443** (and 80 for the HTTPS redirect) to nginx.
- Do **not** publish SurvNG’s own port (`8088` by default) to the internet. Bind it to `127.0.0.1` if nginx is local:

```text
python -m survng.app --host 127.0.0.1 --port 8088
```

The packaged systemd unit listens on `0.0.0.0:8088` so cameras on the LAN can open the console. For internet use, change `--host 0.0.0.0` to `--host 127.0.0.1` and keep 443 on nginx. Reload systemd after editing the unit.

## 6. What SurvNG then enforces

- Session cookie is HTTP-only, `SameSite=Lax`, **Secure** on HTTPS, and scoped to the web base path.
- Changing a user’s password signs out their other browsers.
- Login is rate-limited per username and per client IP.
- Cross-origin API calls are rejected.
- Password hashes use scrypt; oversized hash parameters are rejected.
- Certificate uploads are limited to 256 KB.

API **bearer tokens** (Home Assistant) remain a separate integration path. Give tokens the smallest scope they need, and prefer them on the LAN or VPN.

If the browser origin is not the Host SurvNG sees (uncommon with the nginx example above), set `SURVNG_TRUSTED_ORIGINS` to the public origin, for example `https://survng.example.com`. Leave it unset when Host and HTTPS already match the URL people type.

## Related

- [Access](access.md)
- [Getting started](getting-started.md)
- [HTTP API](api.md)
- [Docker installation](../docker.md)
