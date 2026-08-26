# Getting started

This walkthrough assumes SurvNG is already installed and reachable in a browser.
If you still need to install it, use the repository files `README.install`
([Docker](https://github.com/InstigatorX/SurvNG/blob/v1.1/README.install.docker.md)
or
[systemd](https://github.com/InstigatorX/SurvNG/blob/v1.1/README.install.systemd.md)),
or the [Docker installation notes](../docker.md), then return here.

## 1. Open SurvNG

In a browser, open the address your installer printed. A common local address is:

```text
http://127.0.0.1:8088/survng/
```

On another device on the same network, use the server’s LAN address instead of `127.0.0.1`.

You should land on **Live**. The left side lists the main workspaces: Live, Incidents, Timeline, Exports, Search, People, and Admin.

If the host already requires sign-in, use the account an administrator created for you.

To lock a new install, open **Admin → Access**, enable sign-in, Save settings, and create the administrator if prompted. Details: [Access](access.md).

If this host will be reachable from the internet, turn sign-in on **before** opening a public port, and put nginx (or another proxy) in front. Step-by-step: [Reverse proxy](reverse-proxy.md).

![Live camera grid](images/live-command-center.png)

## 2. Check that the server looks healthy

Glance at the top status strip. Healthy usually means SurvNG is running and can see its storage. If cameras are not configured yet, that is normal.

Open **Admin → Health** if you want a fuller picture later.

## 3. Add your first camera

1. Open **Admin → Cameras**.
2. Add a camera and give it a clear name such as `Front Door`.
3. Paste the camera’s **main** video URL into the main stream field.
4. If the camera has a lighter **sub** stream, paste that into the live stream field.
5. If the camera supports ONVIF motion notices, fill in the ONVIF host, port, username, and password.
6. Save.

![Camera settings in Admin](images/admin-cameras.png)

**Example Reolink-style RTSP links:**

```text
rtsp://USER:PASSWORD@CAMERA_IP:554/h264Preview_01_main
rtsp://USER:PASSWORD@CAMERA_IP:554/h264Preview_01_sub
```

Many Reolink cameras use ONVIF on port `8000`, but check the camera’s own network settings.

More detail: [Cameras](cameras.md).

## 4. Confirm live video

Return to **Live**. You should see a tile for the new camera. If the picture is dark or frozen:

- Confirm the stream URL works in another player (for example VLC).
- Confirm the camera and SurvNG are on a reachable network path.
- Check **Admin → Cameras → Info** for connection notes.

## 5. Confirm recording

Leave the camera enabled for a few minutes, then open **Timeline**, select the camera, and scrub near “now.” You should see continuous video segments filling in.

![Timeline workspace](images/timeline-workspace.png)

If Timeline is empty, confirm recording is enabled for that camera under **Admin → Cameras**.

## 6. Choose a motion behavior

Under **Admin → Cameras → Motion/Object**, pick a motion behavior. The recommended default is **Camera + EMA backup**:

- The camera’s own motion notice is primary.
- SurvNG’s visual backup can still catch activity if a notice is missed.

If your camera never sends reliable motion notices, choose **EMA only**.

Overview: [Motion & detection](motion-detection.md).

## 7. Optional: turn on object detection

Object detection is optional but useful. Under **Admin → Detection**:

1. Enable the detector.
2. Point SurvNG at an OpenVINO-readable model and label file.
3. Start with the default confidence settings.
4. Watch **Incidents** after walking through a watched area.

Without a model, SurvNG still records and can store motion-related notes; it just will not label people or vehicles.

## 8. Draw a zone

Open the camera’s **Zones** tab and draw a polygon over the area you care about — a doorway, driveway apron, or gate — rather than an entire busy street if that creates noise.

See [Zones](zones.md).

## 9. Explore the rest of the product

| Goal | Where to go |
| --- | --- |
| Watch right now | [Live](live.md) |
| Review kept activity | [Incidents](incidents.md) |
| Scrub recorded video | [Timeline](timeline.md) |
| Save a clip | [Timeline & exports](timeline.md) |
| Find “person in a red jacket” | [Search](search.md) |
| Name a recurring face | [People](people.md) |
| Ask about an incident | [AI assistant](assistant.md) |

## Security reminder

On a private LAN you can leave sign-in off. If this host will be on the internet, turn sign-in on first, put HTTPS on nginx (or another reverse proxy), and do not publish port `8088`. Full steps: [Reverse proxy](reverse-proxy.md).
