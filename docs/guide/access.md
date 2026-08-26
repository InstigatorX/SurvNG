# Access

**Access** covers who can open SurvNG in a browser and whether the console is served over HTTPS.

Sign-in is optional and off by default, matching SurvNG’s LAN-first install. When you are ready to lock the console, require sign-in and create an administrator.

## Roles

| Role | Can do |
| --- | --- |
| **Admin** | Everything in the browser, including cameras, detection, storage, users, and HTTPS |
| **Viewer** | Live, Incidents, Timeline, Exports, Search, and People. Admin and the assistant are hidden |

Home Assistant and other automations still use **API tokens** under [Integrations](integrations.md). Those bearer tokens are separate from browser users.

## Turn on sign-in

1. Open **Admin → Access**.
2. Enable **Require sign-in for the browser console**.
3. **Save settings**.
4. If no users exist yet, SurvNG asks you to **create the administrator**. Otherwise, sign in with an existing account.

You can also add Admin and Viewer accounts on the Access page before enabling sign-in. User create, role, password, and delete still apply immediately; requiring sign-in and session length wait for Save.

The next visit shows the sign-in screen. Sessions stay on the device as an HTTP-only cookie for the number of days set under **Session length** (default 14). Changing that length and saving applies to the next sign-in.

To go back to an open LAN console, turn off **Require sign-in** and Save, then demote or delete accounts. Deleting the last administrator also turns sign-in off. While sign-in is required, the last administrator cannot be changed to Viewer.

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
