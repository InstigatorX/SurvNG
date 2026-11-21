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
  const text = await page.locator(".recording-hero-controls time").first().textContent();
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
    const hero = /(\d+:\d\d:\d\d)/.exec(document.querySelector(".recording-hero-controls time")?.textContent || "")?.[1];
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
  await lanes.locator(".catchup-event button", { hasText: "Person" }).first().click();
  await waitParam(lanes, "event", "value !== null", "a story card plays its event");
  assert.deepEqual(lanes.errors, [], "lanes and catch-up interactions must not raise errors");
  await lanes.close();

  // Desktop day navigation lives in the rail controls.
  const days = await open({ width: 1440, height: 900 }, query);
  await settle(days);
  for (const name of ["Previous day", "Next day"]) assert.ok(await days.getByRole("button", { name, exact: true }).first().isVisible(), `${name} is visible on desktop`);
  assert.ok(await days.getByRole("button", { name: "Today" }).isVisible(), "Today is visible on desktop");
  assert.ok(await days.locator(".rail-day-toolbar .recordings-playback-rate select").isVisible(), "playback speed is visible on desktop");
  await days.locator(".rail-day-toolbar").getByRole("button", { name: "Previous day" }).click();
  await waitParam(days, "date", "value === '2026-09-08'", "Previous day changes the recording day");
  await days.locator(".rail-day-toolbar").getByRole("button", { name: "Next day" }).click();
  await waitParam(days, "date", "value === '2026-09-09'", "Next day changes the recording day");
  assert.deepEqual(days.errors, [], "day navigation must not raise errors");
  await days.close();

  // iPhone playback transport: per-file MP4 by default; ?transport=hls forces
  // the HLS playlist (remembered per browser) and shows the diagnostics readout.
  const iphone = { userAgent: "Mozilla/5.0 (iPhone; CPU iPhone OS 18_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Mobile/15E148 Safari/604.1" };
  const phone = { width: 390, height: 844 };
  const readout = (page) => page.locator(".playback-diagnostics strong").textContent();
  const defaultIphone = await open(phone, `${query}&playbackdebug=1`, true, iphone);
  await defaultIphone.waitForSelector(".playback-diagnostics");
  await defaultIphone.waitForFunction(() => /per-file MP4/.test(document.querySelector(".playback-diagnostics strong")?.textContent || ""));
  assert.match(await readout(defaultIphone), /^original · per-file MP4/, "an iPhone plays original files by default");
  await defaultIphone.close();
  const forced = await open(phone, `${query}&transport=hls`, true, iphone);
  await forced.waitForSelector(".playback-diagnostics");
  await forced.waitForFunction(() => /^hls \(forced\)/.test(document.querySelector(".playback-diagnostics strong")?.textContent || ""));
  await forced.waitForFunction(() => document.querySelector(".recording-hero-media video")?.readyState >= 2);
  await forced.waitForTimeout(1500);
  assert.match(await readout(forced), /^hls \(forced\)/, "forced HLS stays on HLS without falling back");
  assert.equal(await forced.locator(".playback-diagnostics-error").count(), 0, "forced HLS plays the fixture without media errors");
  assert.equal(await forced.evaluate(() => [...document.querySelectorAll(".recording-hero-media video")].some((video) => /segment\.mp4/.test(video.getAttribute("src") || ""))), false, "forced HLS does not load per-file MP4");
  await forced.goto(forced.url().replace("&transport=hls", ""));
  await forced.waitForFunction(() => /^hls \(forced\)/.test(document.querySelector(".playback-diagnostics strong")?.textContent || ""));
  await forced.goto(`${forced.url()}&transport=auto`);
  await forced.waitForSelector(".recording-hero-media video");
  await forced.waitForTimeout(500);
  assert.equal(await forced.locator(".playback-diagnostics").count(), 0, "transport=auto clears the forced choice and its readout");
  assert.deepEqual(forced.errors, [], "forced transport pages must not raise errors");
  await forced.close();

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
  await mobile.waitForSelector(".edge-rail-loupe img");
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
  assert.equal(await cameraSelect.locator("option").count(), 3, "every camera is selectable on a phone");
  await cameraSelect.selectOption("front-door");
  await waitParam(mobile, "camera", "value === 'front-door'", "the phone camera selector switches camera");
  await mobile.locator(".edge-rail").getByRole("button", { name: "Previous day" }).click();
  await waitParam(mobile, "date", "value === '2026-09-08'", "the edge rail steps to the previous day");
  await mobile.locator(".edge-rail").getByRole("button", { name: "Next day" }).click();
  await waitParam(mobile, "date", "value === '2026-09-09'", "the edge rail steps to the next day");
  assert.deepEqual(mobile.errors, [], "mobile interactions must not raise errors");
  await mobile.close();
}
