// Checks for tests/timeline-rail.mjs; kept separate so the fixture stays small.
async function settle(page) {
  await page.waitForSelector(".recordings-v2-page");
  await page.waitForTimeout(500);
}
const param = (page, name) => new URL(page.url()).searchParams.get(name);
async function waitParam(page, name, predicate, message) {
  await page.waitForFunction(([key, source]) => (new Function("value", `return ${source}`))(new URL(location.href).searchParams.get(key)), [name, predicate], { timeout: 5000 })
    .catch(() => { throw new Error(`${message} (${name}=${new URL(page.url()).searchParams.get(name)})`); });
}

// The playhead the page reports, as seconds since midnight (fixture is UTC).
// The rail's clock at its playhead line (h:mm:ss, no meridiem; this fixture only scrubs PM times).
async function railSeconds(page) {
  const text = await page.locator(".event-rail .event-rail-playhead span").textContent();
  const [h, m, sec] = text.trim().split(":").map(Number);
  return (h % 12 + 12) * 3600 + m * 60 + (sec || 0);
}

async function playheadSeconds(page) {
  const text = await page.locator(".recording-hero-controls :is(time, .recordings-v2-date-toggle)").first().textContent();
  const match = /(\d+):(\d\d):(\d\d)\s*([AP]M)/.exec(text || "");
  if (!match) return NaN;
  const hours = (Number(match[1]) % 12) + (match[4] === "PM" ? 12 : 0);
  return hours * 3600 + Number(match[2]) * 60 + Number(match[3]);
}
async function waitPlayhead(page, predicate, message) {
  const deadline = Date.now() + 6000;
  let value = NaN;
  while (Date.now() < deadline) {
    value = await playheadSeconds(page);
    if (predicate(value)) return value;
    await page.waitForTimeout(150);
  }
  throw new Error(`${message} (playhead ${value}s)`);
}

export async function checks({ open, shot, query, at, assert }) {
  // Screenshots of every view in both form factors and both themes.
  for (const [name, viewport, mobile] of [["desktop", { width: 1440, height: 900 }, false], ["mobile", { width: 390, height: 844 }, true]]) {
    for (const view of ["rail", "cameras", "catchup"]) {
      for (const theme of ["dark", "light"]) {
        const page = await open(viewport, `${query}&view=${view}&theme=${theme}`, mobile);
        await settle(page);
        await shot(page, `${name}-${view}${theme === "light" ? "-light" : ""}`);
        assert.deepEqual(page.errors, [], `${name} ${view} ${theme} must not raise errors`);
        await page.close();
      }
    }
  }

  // A · Event Rail on desktop.
  const desktop = await open({ width: 1440, height: 900 }, query);
  await settle(desktop);
  const scroller = desktop.locator(".event-rail .event-rail-scroller");
  assert.match(await desktop.locator(".event-rail .event-rail-playhead span").textContent(), /^2:31/, "rail starts at the requested time");
  assert.equal(await desktop.locator(".recordings-v2-track").count(), 0, "the rail replaces the horizontal track");
  const startAt = await playheadSeconds(desktop);
  assert.ok(Math.abs(startAt - 14.53 * 3600) < 30, `playback starts near 2:31 PM (${startAt})`);
  assert.equal(await railSeconds(desktop), startAt, "the rail line shows the video's own time, not a pixel-rounded one");
  await scroller.hover();
  await desktop.mouse.wheel(0, 360);
  await waitPlayhead(desktop, (value) => value < startAt - 300, "scrolling down the rail seeks earlier");
  await desktop.waitForTimeout(800);
  const afterWheel = await railSeconds(desktop);
  await desktop.waitForTimeout(1200);
  const settled = await railSeconds(desktop);
  assert.ok(settled >= afterWheel - 1 && settled - afterWheel < 5, `a committed scrub is not undone by stale playback time (${afterWheel} → ${settled})`);
  await desktop.waitForFunction(() => {
    const rail = document.querySelector(".event-rail .event-rail-playhead span")?.textContent?.trim();
    const hero = /(\d+:\d\d:\d\d)/.exec(document.querySelector(".recording-hero-controls :is(time, .recordings-v2-date-toggle)")?.textContent || "")?.[1];
    return rail && rail === hero;
  }, null, { timeout: 5000 }).catch(() => { throw new Error("the rail line matches playback after a scrub"); });
  await desktop.locator(".event-rail-card", { hasText: "Car" }).first().click();
  await waitParam(desktop, "event", "value !== null", "selecting a card selects its incident");
  await waitPlayhead(desktop, (value) => Math.abs(value - 14.13 * 3600) < 20, "a card seeks to its event");
  await desktop.waitForFunction(() => /^2:0[78]/.test(document.querySelector(".event-rail-playhead span")?.textContent || ""));
  await desktop.getByRole("group", { name: "Rail zoom" }).getByRole("button", { name: "15m" }).click();
  await waitParam(desktop, "rail", "value === '15'", "rail zoom persists");
  await desktop.getByRole("group", { name: "Event type" }).getByRole("button", { name: "People" }).click();
  await waitParam(desktop, "filter", "value === 'people'", "rail filter persists");
  assert.equal(await desktop.locator(".event-rail-card.vehicle").count(), 0, "the People filter hides vehicles");
  await desktop.keyboard.press("Tab");
  await scroller.focus();
  const beforeKey = await playheadSeconds(desktop);
  await desktop.keyboard.press("PageDown");
  await waitPlayhead(desktop, (value) => value < beforeKey - 60, "PageDown scrubs earlier from the keyboard");

  await desktop.getByRole("group", { name: "Timeline view" }).getByRole("button", { name: "Cameras" }).click();
  await waitParam(desktop, "view", "value === 'cameras'", "the view switch persists");
  await desktop.waitForSelector(".camera-lane");
  // Back restores the screen the URL describes, not just the URL.
  await desktop.goBack();
  await waitParam(desktop, "view", "value === null", "Back returns to the rail URL");
  await desktop.waitForSelector(".event-rail-card");
  assert.equal(await desktop.locator(".camera-lane").count(), 0, "Back leaves the camera lanes");
  await desktop.goBack();
  await desktop.goBack();
  await waitParam(desktop, "rail", "value === null", "Back undoes the rail zoom");
  await desktop.waitForFunction(() => document.querySelector('[aria-label="Rail zoom"] button[aria-pressed="true"]')?.textContent === "1h");
  await desktop.goForward();
  await desktop.goForward();
  await desktop.waitForFunction(() => document.querySelector('[aria-label="Rail zoom"] button[aria-pressed="true"]')?.textContent === "15m");
  assert.deepEqual(desktop.errors, [], "desktop rail interactions must not raise errors");
  await desktop.close();

  // C · Camera lanes, from a fresh page at 2:31 PM.
  // On Sub, whose recording has a gap across 2:31 PM that Main does not.
  const lanes = await open({ width: 1440, height: 900 }, `${query.replace("source=main", "source=live")}&view=cameras`);
  await settle(lanes);
  await lanes.waitForSelector(".camera-lane");
  assert.equal(await lanes.locator(".camera-lane").count(), 3, "one lane per camera");
  const coveredAtPlayhead = await lanes.evaluate(() => {
    const x = document.querySelector(".camera-lanes-playhead").getBoundingClientRect().x;
    return [...document.querySelector(".camera-lane").querySelectorAll(".camera-lane-coverage")].some((el) => { const r = el.getBoundingClientRect(); return r.x <= x && x <= r.right; });
  });
  assert.ok(!coveredAtPlayhead, "lanes draw the selected Sub source, not Main");
  assert.match(await lanes.locator(".camera-lanes-link").first().textContent(), /Same person · Gate → Front Door/);
  await lanes.locator(".camera-lane-name", { hasText: "Front Door" }).click();
  await waitParam(lanes, "camera", "value === 'front-door'", "a lane opens its camera");

  // D · Catch-up.
  await lanes.getByRole("group", { name: "Timeline view" }).getByRole("button", { name: "Catch up" }).click();
  await lanes.waitForSelector(".catchup-event");
  assert.ok(await lanes.locator(".catchup-quiet").count() > 0, "quiet stretches collapse");
  assert.equal(await lanes.getByRole("button", { name: "Auto-advance" }).count(), 0, "Catch up always advances; there is no toggle");
  await lanes.locator(".catchup-event button", { hasText: "Person" }).first().click();
  await waitParam(lanes, "event", "value !== null", "a story card plays its event");
  assert.deepEqual(lanes.errors, [], "lanes and catch-up interactions must not raise errors");
  await lanes.close();

  // Desktop day navigation lives in the player HUD: previous/next day on
  // either side of the date picker; Today and speed beside the transport.
  const days = await open({ width: 1440, height: 900 }, query);
  await settle(days);
  const hud = days.locator(".recording-hero-controls");
  for (const name of ["Previous day", "Next day", "Previous event", "Next event"]) assert.ok(await hud.getByRole("button", { name, exact: true }).isVisible(), `${name} is in the HUD on desktop`);
  for (const name of ["Today", "Nearby", "Export"]) assert.ok(await hud.getByRole("button", { name }).isVisible(), `${name} is in the HUD on desktop`);
  assert.ok(await hud.locator(".timeline-hud-rate select").isVisible(), "playback speed is in the HUD on desktop");
  assert.equal(await hud.locator(".timeline-hud-day .recordings-v2-date-toggle").count(), 1, "the HUD date opens the day picker");
  assert.equal(await days.locator(".rail-controls").isVisible(), false, "no control row under the player");
  await hud.getByRole("button", { name: "Previous day", exact: true }).click();
  await waitParam(days, "date", "value === '2026-09-08'", "Previous day changes the recording day");
  await days.locator(".recording-hero-controls").getByRole("button", { name: "Next day", exact: true }).click();
  await waitParam(days, "date", "value === '2026-09-09'", "Next day changes the recording day");
  assert.deepEqual(days.errors, [], "day navigation must not raise errors");
  await days.close();

  // iPhone defaults to HLS. Explicit diagnostics settings take precedence over
  // the readout that accompanies a forced transport, including saved overrides.
  const iphone = { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1" };
  const phone = { width: 390, height: 844 };
  const readout = (page) => page.locator(".playback-diagnostics strong").textContent();
  const defaultIphone = await open(phone, `${query}&playbackdebug=1`, true, iphone);
  await defaultIphone.waitForSelector(".playback-diagnostics");
  await defaultIphone.waitForFunction(() => document.querySelector(".recording-hero-media video")?.readyState >= 2);
  assert.match(await readout(defaultIphone), /^hls ·/, "an iPhone uses HLS by default");
  await defaultIphone.close();
  const forced = await open(phone, `${query}&transport=hls`, true, iphone);
  await forced.waitForSelector(".playback-diagnostics");
  await forced.waitForFunction(() => /^hls \(forced\)/.test(document.querySelector(".playback-diagnostics strong")?.textContent || ""));
  await forced.waitForFunction(() => document.querySelector(".recording-hero-media video")?.readyState >= 2);
  await forced.waitForTimeout(1500);
  assert.match(await readout(forced), /^hls \(forced\)/, "forced HLS stays on HLS without falling back");
  assert.equal(await forced.locator(".playback-diagnostics-error").count(), 0, "forced HLS plays the fixture without media errors");
  assert.equal(await forced.evaluate(() => [...document.querySelectorAll(".recording-hero-media video")].some((video) => /segment\.mp4/.test(video.getAttribute("src") || ""))), false, "forced HLS does not load per-file MP4");
  const unforcedUrl = forced.url();
  const forcedUrl = new URL(unforcedUrl);
  forcedUrl.searchParams.set("transport", "hls");
  for (const url of [forcedUrl, unforcedUrl]) {
    await forced.goto(`${url}&playbackdebug=0`);
    await forced.waitForFunction(() => document.querySelector(".recording-hero-media video")?.readyState >= 2);
    assert.equal(await forced.locator(".playback-diagnostics").count(), 0, "playbackdebug=0 hides diagnostics with an explicit or saved override");
    assert.equal(param(forced, "playbackdebug"), "0", "Timeline preserves the explicit debug setting in the URL");
  }
  await forced.reload();
  await forced.waitForFunction(() => document.querySelector(".recording-hero-media video")?.readyState >= 2);
  assert.equal(await forced.locator(".playback-diagnostics").count(), 0, "diagnostics stay off on reload");
  await forced.goto(unforcedUrl);
  await forced.waitForFunction(() => /^hls \(forced\)/.test(document.querySelector(".playback-diagnostics strong")?.textContent || ""));
  await forced.goto(`${forced.url()}&transport=auto`);
  await forced.waitForSelector(".recording-hero-media video");
  await forced.waitForTimeout(500);
  assert.equal(await forced.locator(".playback-diagnostics").count(), 0, "transport=auto clears the forced choice and its readout");
  assert.deepEqual(forced.errors, [], "forced transport pages must not raise errors");
  await forced.close();

  // A selected event in the URL (a shared link, reload or back/forward) must not
  // open the incident panel over the phone rail and squeeze it to its controls.
  const linked = await open({ width: 390, height: 844 }, `${query}&event=8`, true);
  await settle(linked);
  for (const pass of ["opened", "reloaded"]) {
    if (pass === "reloaded") { await linked.reload(); await settle(linked); }
    assert.match(await linked.locator(".recordings-v2-page").getAttribute("class"), /has-investigation/, `the event link selects its incident (${pass})`);
    const railHeight = await linked.locator(".event-rail-scroller").evaluate((element) => element.getBoundingClientRect().height);
    assert.ok(railHeight > 200, `the phone rail keeps its height with an event in the URL (${pass}: ${railHeight}px)`);
    assert.equal(await linked.locator(".recordings-v2-incidents").isVisible(), false, `the phone rail view does not open the incident panel (${pass})`);
  }
  assert.deepEqual(linked.errors, [], "event links on the phone rail must not raise errors");
  await linked.close();

  // Classic layout remains available.
  const classic = await open({ width: 1440, height: 900 }, `${query}&view=classic`);
  await settle(classic);
  assert.equal(await classic.locator(".recordings-v2-track").count(), 1, "classic keeps the horizontal track");
  assert.equal(await classic.locator(".event-rail").count(), 0);
  await classic.close();

  // B · Edge rail with loupe on a phone.
  const mobile = await open({ width: 390, height: 844 }, query, true);
  await settle(mobile);
  const track = mobile.locator(".edge-rail-track");
  const box = await track.boundingBox();
  assert.ok(box && box.height > 200, "the edge rail is tall enough to scrub");
  assert.equal(await railSeconds(mobile), await playheadSeconds(mobile), "the phone rail line shows the video's own time");
  const before = await playheadSeconds(mobile);
  await mobile.mouse.move(box.x + box.width / 2, box.y + box.height * 0.3);
  await mobile.mouse.down();
  await mobile.mouse.move(box.x + box.width / 2, box.y + box.height * 0.6, { steps: 6 });
  await mobile.waitForSelector(".edge-rail-loupe");
  // The scrubbed frame is shown in the player, so the loupe carries only the time.
  assert.equal(await mobile.locator(".edge-rail-loupe img").count(), 0, "the loupe shows no image of its own");
  await shot(mobile, "mobile-loupe");
  await mobile.mouse.up();
  assert.equal(await mobile.locator(".edge-rail-loupe").count(), 0, "the loupe closes on release");
  await waitPlayhead(mobile, (value) => Math.abs(value - before) > 600, "releasing the edge rail seeks");
  // Phones pick the camera from the command bar.
  const cameraSelect = mobile.getByLabel("Timeline camera", { exact: true });
  assert.ok(await cameraSelect.isVisible(), "the phone command bar has a camera selector");
  const bar = await mobile.locator(".recordings-commandbar").boundingBox();
  const selectBox = await cameraSelect.boundingBox();
  assert.ok(selectBox.y >= bar.y && selectBox.y + selectBox.height <= bar.y + bar.height && selectBox.width >= 100, "the camera selector fits the command bar");
  assert.deepEqual(await cameraSelect.locator("option").allTextContents(), ["Gate", "Front Door", "Upper Garage"], "every camera is selectable on a phone, and All cameras is not offered");
  await cameraSelect.selectOption("front-door");
  await waitParam(mobile, "camera", "value === 'front-door'", "the phone camera selector switches camera");
  await mobile.locator(".edge-rail").getByRole("button", { name: "Previous day" }).click();
  await waitParam(mobile, "date", "value === '2026-09-08'", "the edge rail steps to the previous day");
  await mobile.locator(".edge-rail").getByRole("button", { name: "Next day" }).click();
  await waitParam(mobile, "date", "value === '2026-09-09'", "the edge rail steps to the next day");
  assert.deepEqual(mobile.errors, [], "mobile interactions must not raise errors");
  await mobile.close();

  // All cameras (desktop and tablet): a grid of the cameras in the player's
  // spot, driven by the same playhead, laid out like Live. Tapping a tile
  // focuses it (Live's Focus) without reloading any tile's video.
  {
    const grid = await open({ width: 1440, height: 900 }, query.replace("camera=gate", "camera=all"));
    await settle(grid);
    await grid.waitForFunction(() => [...document.querySelectorAll(".all-cameras-tile video")].filter((video) => video.readyState >= 2).length === 3, null, { timeout: 8000 })
      .catch(() => { throw new Error("every All cameras tile loads its recording"); });
    assert.deepEqual(await grid.locator(".all-cameras-tile").evaluateAll((tiles) => tiles.map((tile) => tile.dataset.cameraId)), ["gate", "front-door", "upper-garage"]);
    assert.equal(await grid.locator(".recording-hero-media > video").count(), 0, "the single-camera player is replaced, not stacked");
    const media = await grid.locator(".all-cameras-media").boundingBox();
    const tiles = await grid.locator(".all-cameras-tile").evaluateAll((elements) => elements.map((element) => element.getBoundingClientRect().toJSON()));
    assert.ok(tiles.every((tile) => tile.y >= media.y - 1 && tile.y + tile.height <= media.y + media.height + 1 && tile.x >= media.x - 1 && tile.x + tile.width <= media.x + media.width + 1 && tile.height > 60), "tiles fill the player area");
    const controls = await grid.locator(".all-cameras-controls").boundingBox();
    assert.ok(tiles.every((tile) => tile.y + tile.height <= controls.y + 1), "the grid controls do not cover the tiles");
    assert.ok((await grid.locator(".event-rail-card small").first().textContent()).includes(" · "), "rail cards name their camera");
    // Live's controls, in Live's order, on the right of the command bar.
    const bar = grid.locator(".recordings-commandbar");
    assert.deepEqual((await bar.locator(".live-layout-control > button").allTextContents()).map((text) => text.trim()), ["Automatic", "Focus", "Custom"]);
    assert.ok(await bar.getByRole("button", { name: "Automatic", exact: true }).getAttribute("aria-pressed") === "true");
    assert.deepEqual((await bar.locator(".live-density-control > button").allTextContents()).map((text) => text.trim()), ["", "4", "6", "9", "16", "25"]);
    const barBox = await bar.boundingBox();
    const layoutBox = await bar.locator(".timeline-grid-layout").boundingBox();
    assert.ok(layoutBox.y >= barBox.y && layoutBox.y + layoutBox.height <= barBox.y + barBox.height && layoutBox.x + layoutBox.width <= barBox.x + barBox.width + 1, "the layout controls fit in the command bar");
    await grid.evaluate(() => { document.querySelectorAll(".all-cameras-tile video").forEach((video, index) => { video.dataset.mount = String(index); }); });
    const front = grid.locator('.all-cameras-tile[data-camera-id="front-door"]');
    await front.locator(".all-cameras-tile-hit").click();
    await grid.waitForSelector('.all-cameras-tile.primary[data-camera-id="front-door"]');
    assert.ok(await bar.getByRole("button", { name: "Focus", exact: true }).getAttribute("aria-pressed") === "true", "tapping a tile switches to Focus");
    assert.deepEqual(await grid.locator(".all-cameras-tile video").evaluateAll((videos) => videos.map((video) => video.dataset.mount)), ["0", "1", "2"], "focusing keeps every tile's video element");
    const primary = await front.boundingBox();
    const other = await grid.locator('.all-cameras-tile[data-camera-id="gate"]').boundingBox();
    assert.ok(primary.width > other.width * 1.5 && primary.height > other.height * 1.5, "the focused camera is the large tile of the mosaic");
    assert.equal(await front.locator(".all-cameras-tile-hit").evaluate((button) => getComputedStyle(button).backgroundColor), "rgba(0, 0, 0, 0)", "the pressed, hovered tile stays see-through");
    await shot(grid, "all-cameras-desktop");
    await bar.getByRole("button", { name: "Automatic", exact: true }).click();
    assert.equal(await grid.locator(".all-cameras-tile.primary").count(), 0, "Automatic returns to the uniform grid");
    const exportButton = grid.getByRole("button", { name: "Select camera" });
    assert.ok(await exportButton.isDisabled(), "export needs one camera");
    const before = await grid.locator(".all-cameras-tile video").evaluateAll((videos) => videos.map((video) => video.currentTime));
    await grid.getByRole("button", { name: "Back 10 seconds" }).click();
    await grid.waitForFunction((previous) => [...document.querySelectorAll(".all-cameras-tile video")].every((video, index) => Math.abs(video.currentTime - (previous[index] - 10)) < 1.5), before, { timeout: 5000 })
      .catch(() => { throw new Error("skipping moves every tile with the playhead"); });
    assert.equal(param(grid, "camera"), "all", "the URL keeps All cameras");
    assert.equal(await grid.locator("img.all-cameras-still").count(), 0, "tiles showing video stop requesting stills");
    // Play/Pause is a choice that survives the index reload a Main/Sub flip makes.
    const transport = grid.locator(".all-cameras-controls");
    const flip = async (name, value) => {
      await transport.getByRole("button", { name, exact: true }).click();
      await waitParam(grid, "source", `value === '${value}'`, `the grid switches to ${name}`);
      await grid.waitForFunction(() => document.querySelectorAll(".all-cameras-tile video").length === 3, null, { timeout: 8000 });
      await grid.waitForTimeout(1500);
    };
    await transport.getByRole("button", { name: "Play", exact: true }).click();
    await flip("Sub", "live");
    assert.ok(await transport.getByRole("button", { name: "Pause", exact: true }).isVisible(), "a playing grid keeps playing after Main/Sub");
    await transport.getByRole("button", { name: "Pause", exact: true }).click();
    await flip("Main", "main");
    assert.ok(await transport.getByRole("button", { name: "Play", exact: true }).isVisible(), "a paused grid stays paused after Main/Sub");
    // Custom: the chosen cameras, kept across visits; Reset shows every camera.
    await bar.getByRole("button", { name: "Custom", exact: true }).click();
    assert.equal(await bar.locator(".live-density-control").count(), 0, "density only applies to Automatic");
    await bar.getByRole("button", { name: /Choose cameras/ }).click();
    await bar.getByRole("checkbox", { name: /Front Door/ }).click();
    await grid.keyboard.press("Escape");
    await grid.waitForFunction(() => document.querySelectorAll(".all-cameras-tile").length === 2, null, { timeout: 5000 })
      .catch(() => { throw new Error("Custom shows only the chosen cameras"); });
    assert.match(await grid.locator(".all-cameras-summary").textContent(), /^2 of 2 cameras$/);
    await shot(grid, "all-cameras-custom");
    await grid.reload();
    await settle(grid);
    await grid.waitForSelector(".all-cameras-tile");
    assert.deepEqual(await grid.locator(".all-cameras-tile").evaluateAll((tiles) => tiles.map((tile) => tile.dataset.cameraId)), ["gate", "upper-garage"], "the Custom set and mode are remembered");
    await grid.locator('.all-cameras-tile[data-camera-id="upper-garage"] .all-cameras-tile-hit').click();
    await grid.waitForSelector('.all-cameras-tile.primary[data-camera-id="upper-garage"]');
    assert.deepEqual(await grid.locator(".all-cameras-tile").evaluateAll((tiles) => tiles.map((tile) => tile.dataset.cameraId)), ["gate", "upper-garage"], "Focus entered from Custom keeps the chosen cameras");
    await bar.getByRole("button", { name: "Custom", exact: true }).click();
    await bar.getByRole("button", { name: /Choose cameras/ }).click();
    await bar.getByRole("checkbox", { name: /Gate/ }).click();
    assert.ok(await bar.getByRole("checkbox", { name: /Upper Garage/ }).isDisabled(), "the last chosen camera cannot be cleared");
    await grid.keyboard.press("Escape");
    await bar.getByRole("button", { name: "Reset", exact: true }).click();
    await grid.waitForFunction(() => document.querySelectorAll(".all-cameras-tile").length === 3, null, { timeout: 5000 })
      .catch(() => { throw new Error("Reset shows every camera"); });
    assert.deepEqual(grid.errors, [], "All cameras must not raise errors");
    await grid.close();
  }

  // Phones never show All cameras: a camera=all link opens the first camera.
  {
    const phone = await open({ width: 390, height: 844 }, query.replace("camera=gate", "camera=all"), true);
    await settle(phone);
    await waitParam(phone, "camera", "value === 'gate'", "a phone opens the first camera instead of All cameras");
    assert.equal(await phone.locator(".all-cameras-grid").count(), 0);
    assert.equal(await phone.locator(".timeline-grid-layout").count(), 0, "phones get no grid layout controls");
    assert.deepEqual(phone.errors, [], "a camera=all link on a phone must not raise errors");
    await phone.close();
  }

  // Cameras with no recording in the playhead's window get no tile; the
  // control bar says which, centred between the time and Main/Sub.
  {
    const grid = await open({ width: 1440, height: 900 }, query.replace("camera=gate", "camera=all"));
    await grid.route("**/api/recordings/grid/day*", async (route) => {
      const response = await route.fetch();
      const payload = await response.json();
      payload.availability = payload.availability.filter((range) => range.camera_id !== "upper-garage");
      await route.fulfill({ response, json: payload });
    });
    await grid.reload();
    await settle(grid);
    await grid.waitForFunction(() => document.querySelectorAll(".all-cameras-tile").length === 2, null, { timeout: 8000 })
      .catch(() => { throw new Error("a camera without recording gets no tile"); });
    assert.deepEqual(await grid.locator(".all-cameras-tile").evaluateAll((tiles) => tiles.map((tile) => tile.dataset.cameraId)), ["gate", "front-door"]);
    const summary = grid.locator(".all-cameras-summary");
    assert.match(await summary.textContent(), /^2 of 3 cameras · No recording: Upper Garage$/);
    const bar = await grid.locator(".all-cameras-controls").boundingBox();
    const box = await summary.boundingBox();
    const source = await grid.locator(".all-cameras-controls .incident-recording-source").boundingBox();
    assert.ok(box.width > 40 && box.x + box.width <= source.x + 1 && box.y >= bar.y && box.y + box.height <= bar.y + bar.height, "the status sits in the bar before Main/Sub");
    await shot(grid, "all-cameras-hidden-desktop");
    assert.deepEqual(grid.errors, [], "hiding cameras must not raise errors");
    await grid.close();
  }

  // Today, the grid starts behind live so tiles have footage ahead of them.
  {
    const liveEdge = at(15.4);
    const grid = await open({ width: 1440, height: 900 }, `camera=all&date=${new Date(liveEdge * 1000).toISOString().slice(0, 10)}`);
    await grid.clock.setFixedTime(new Date((liveEdge + 20) * 1000));
    await grid.reload();
    await settle(grid);
    await grid.waitForSelector(".all-cameras-tile video");
    const start = await playheadSeconds(grid);
    const edge = liveEdge - at(0);
    assert.ok(start <= edge - 45 && start >= edge - 400, `today starts about a minute behind the newest recording (${edge - start}s behind)`);
    await grid.close();
  }
}
