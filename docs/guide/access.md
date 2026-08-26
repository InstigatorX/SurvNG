# Access

**Access** covers who can open SurvNG in a browser and whether the console is served over HTTPS.

Sign-in is optional and off by default, matching SurvNG’s LAN-first install. When you are ready to lock the console, create an administrator and require sign-in.

## Roles

| Role | Can do |
| --- | --- |
| **Admin** | Everything in the browser, including cameras, detection, storage, users, and HTTPS |
| **Viewer** | Live, Incidents, Timeline, Exports, Search, and People. Admin and the assistant are hidden |

Home Assistant and other automations still use **API tokens** under [Integrations](integrations.md). Those bearer tokens are separate from browser users.

## Turn on sign-in

1. Open **Admin → Access**.
2. Add at least one **Admin** user.
3. Optionally add **Viewer** accounts for family or staff.
4. Enable **Require sign-in for the browser console**.

The next visit shows the sign-in screen. Sessions last two weeks and stay on the device as an HTTP-only cookie.

Passwords are stored as scrypt hashes. SurvNG never writes the password itself into `config.json`.

## HTTPS

SurvNG can serve the UI over TLS without a reverse proxy.

1. Set a **hostname** such as `survng.local` or the server’s LAN DNS name.
2. **Generate self-signed** for a private LAN, or **upload** a PEM certificate and private key from your CA (files or pasted PEM text).
3. Enable **Serve SurvNG over HTTPS**.
4. **Restart with HTTPS**. The process reads the stored certificate on startup.

A self-signed certificate will warn in the browser until you trust it. That is expected on a home LAN.

Start SurvNG with `python -m survng.app` (or the packaged systemd/Docker command) so TLS settings are applied. Plain `uvicorn survng.app.main:app` does not attach the certificate files.

## Related

- [Admin](admin.md)
- [Integrations](integrations.md)
- [HTTP API](api.md)
