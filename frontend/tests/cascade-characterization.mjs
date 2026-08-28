import assert from "node:assert/strict";
import { chromium } from "playwright";

// This is a characterization harness for stylesheet cleanup.  It intentionally
// checks browser-resolved cascade, inheritance, geometry, and stacking instead
// of screenshot pixels, which are not a stable proof of cascade equivalence.
const BASE_URL = "http://127.0.0.1:8088/survng";
const VIEWPORTS = Object.freeze({
  desktop: { width: 1440, height: 900 },
  tablet: { width: 900, height: 844 },
  mobile: { width: 390, height: 844 },
});
const WORKSPACES = Object.freeze([
  { id: "live", path: "/", label: "Live", selector: ".live-grid" },
  { id: "incidents", path: "/incidents", label: "Incidents", selector: ".incidents-page" },
  { id: "timeline", path: "/timeline", label: "Timeline", selector: ".recordings-v2-page" },
  { id: "exports", path: "/exports", label: "Exports", selector: ".export-center" },
  { id: "search", path: "/search", label: "Search", selector: ".semantic-search-page" },
  { id: "people", path: "/people", label: "People", selector: ".faces-page" },
  { id: "admin", path: "/admin", label: "Admin", selector: ".config-grid" },
]);
const INHERITED_TYPOGRAPHY = Object.freeze({
  fontFamily: '"Inter Variable", Inter, "SF Pro Text", "Segoe UI", ui-sans-serif, system-ui, sans-serif',
  fontSize: "14px",
  lineHeight: "18.9px",
  boxSizing: "border-box",
});
const THEME_FINGERPRINTS = Object.freeze({
  light: Object.freeze({
    tokens: { bg: "#edf1f3", ink: "#152025", line: "#cfd8dc", accent: "#0f766e" },
    body: { color: "rgb(21, 32, 37)", backgroundColor: "rgb(237, 241, 243)" },
    shellBackground: "rgb(231, 236, 239)",
    topbarBackground: "rgba(248, 250, 251, 0.88)",
  }),
  dark: Object.freeze({
    tokens: { bg: "#0b0d10", ink: "#f3f4f6", line: "#262b32", accent: "#4f8fd8" },
    body: { color: "rgb(243, 244, 246)", backgroundColor: "rgb(11, 13, 16)" },
    shellBackground: "rgb(16, 19, 24)",
    topbarBackground: "rgba(12, 14, 17, 0.96)",
  }),
});

function urlFor(path) {
  return path === "/" ? `${BASE_URL}/` : `${BASE_URL}${path}`;
}

async function openWorkspace(page, workspace) {
  await page.goto(urlFor(workspace.path), { waitUntil: "domcontentloaded", timeout: 30_000 });
  await page.getByText("Loading workspace…").waitFor({ state: "hidden", timeout: 30_000 });
  const heading = page.locator(".workspace-content > h1");
  await heading.waitFor({ state: "attached", timeout: 30_000 });
  await page.waitForFunction(() => {
    const topbar = document.querySelector(".topbar");
    return topbar
      && getComputedStyle(document.body).fontFamily.includes("Inter Variable")
      && getComputedStyle(topbar).position === "sticky";
  }, undefined, { timeout: 30_000 });
  assert.match(await heading.textContent(), new RegExp(`SurvNG — ${workspace.label}`));
}

async function characterizeCascade(page, workspace, theme, viewportName) {
  const selector = await page.locator(workspace.selector).count()
    ? workspace.selector
    : ".workspace-content";
  const report = await page.evaluate((targetSelector) => {
    const target = document.querySelector(targetSelector);
    const content = document.querySelector(".workspace-content");
    const heading = content?.querySelector("h1");
    const shell = document.querySelector(".app-shell");
    const topbar = document.querySelector(".topbar");
    if (!target || !content || !heading || !shell || !topbar) return null;

    const describe = (node) => {
      const style = getComputedStyle(node);
      const rect = node.getBoundingClientRect();
      return {
        color: style.color,
        backgroundColor: style.backgroundColor,
        display: style.display,
        position: style.position,
        fontFamily: style.fontFamily,
        fontSize: style.fontSize,
        lineHeight: style.lineHeight,
        boxSizing: style.boxSizing,
        custom: {
          bg: style.getPropertyValue("--bg").trim(),
          ink: style.getPropertyValue("--ink").trim(),
          line: style.getPropertyValue("--line").trim(),
          accent: style.getPropertyValue("--accent").trim(),
        },
        rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
        scroll: { width: node.scrollWidth, height: node.scrollHeight, clientWidth: node.clientWidth, clientHeight: node.clientHeight },
      };
    };
    const stacking = (node) => {
      const layers = [];
      for (let current = node; current instanceof Element; current = current.parentElement) {
        const style = getComputedStyle(current);
        layers.push({
          tag: current.tagName,
          className: typeof current.className === "string" ? current.className : "",
          position: style.position,
          zIndex: style.zIndex,
          transform: style.transform,
          opacity: style.opacity,
          filter: style.filter,
          isolation: style.isolation,
          contain: style.contain,
          willChange: style.willChange,
          overflowX: style.overflowX,
          overflowY: style.overflowY,
        });
      }
      return layers;
    };
    return {
      root: describe(document.documentElement),
      body: describe(document.body),
      content: describe(content),
      shell: describe(shell),
      topbar: describe(topbar),
      target: describe(target),
      heading: describe(heading),
      horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
      targetStacking: stacking(target),
    };
  }, selector);

  assert.ok(report, `${workspace.id}: expected shell and representative target`);
  const expected = THEME_FINGERPRINTS[theme];
  assert.ok(expected, `known theme fingerprint: ${theme}`);
  for (const [name, snapshot] of Object.entries({ root: report.root, body: report.body, content: report.content, shell: report.shell, topbar: report.topbar, target: report.target, heading: report.heading })) {
    assert.ok(snapshot.color && snapshot.fontFamily, `${workspace.id}: ${name} has resolved inherited typography`);
    assert.ok(snapshot.custom.bg && snapshot.custom.ink && snapshot.custom.line && snapshot.custom.accent, `${workspace.id}: ${name} receives root custom properties`);
  }
  // These independent exact assertions catch a cascade change even when it
  // affects both a parent and its child, where equality alone would pass.
  assert.deepEqual(report.root.custom, expected.tokens, `${workspace.id}: ${theme} root tokens remain stable`);
  assert.deepEqual(
    Object.fromEntries(["color", "backgroundColor", ...Object.keys(INHERITED_TYPOGRAPHY)].map((key) => [key, report.body[key]])),
    { ...expected.body, ...INHERITED_TYPOGRAPHY },
    `${workspace.id}: ${theme} body typography and colors remain stable`,
  );
  assert.deepEqual(
    Object.fromEntries(["color", "backgroundColor", ...Object.keys(INHERITED_TYPOGRAPHY)].map((key) => [key, report.content[key]])),
    { color: expected.body.color, backgroundColor: "rgba(0, 0, 0, 0)", ...INHERITED_TYPOGRAPHY },
    `${workspace.id}: ${theme} workspace inherited values remain stable`,
  );
  if (viewportName !== "mobile") {
    assert.deepEqual(
      Object.fromEntries(["display", "position", "backgroundColor"].map((key) => [key, report.shell[key]])),
      { display: "grid", position: "static", backgroundColor: expected.shellBackground },
      `${workspace.id}: ${theme} shell cascade remains stable`,
    );
  }
  assert.deepEqual(
    Object.fromEntries(["display", "position", "backgroundColor"].map((key) => [key, report.topbar[key]])),
    { display: "flex", position: "sticky", backgroundColor: expected.topbarBackground },
    `${workspace.id}: ${theme} topbar cascade remains stable`,
  );
  assert.equal(report.horizontalOverflow, false, `${workspace.id}: page has no horizontal document overflow`);
  assert.ok(report.target.rect.width > 0 && report.target.rect.height > 0, `${workspace.id}: representative target has geometry`);
  assert.ok(report.target.scroll.width >= report.target.scroll.clientWidth, `${workspace.id}: representative target has coherent horizontal geometry`);
  assert.ok(report.targetStacking.some((layer) => layer.className.split(/\s+/).includes("app-shell")), `${workspace.id}: representative target remains inside the shell stacking tree`);
}

async function assertShellStacking(page, viewportName) {
  const shell = page.locator(".app-shell");
  assert.ok(await shell.evaluate((node) => node.getBoundingClientRect().width > 0));
  if (viewportName === "mobile") {
    const nav = page.locator(".mobile-workspace-nav");
    assert.equal(await nav.isVisible(), true);
    const fingerprint = await nav.evaluate((node) => {
      const style = getComputedStyle(node);
      return { position: style.position, zIndex: style.zIndex, overflowX: style.overflowX, overflowY: style.overflowY };
    });
    assert.deepEqual(fingerprint, { position: "fixed", zIndex: "60", overflowX: "visible", overflowY: "visible" });
  } else {
    const sidebar = page.locator(".workspace-sidebar");
    assert.equal(await sidebar.isVisible(), true);
    const fingerprint = await sidebar.evaluate((node) => {
      const style = getComputedStyle(node);
      return { position: style.position, zIndex: style.zIndex, overflowX: style.overflowX, overflowY: style.overflowY };
    });
    assert.equal(fingerprint.position, "sticky");
    assert.equal(fingerprint.zIndex, "20");
  }
}

async function assertMobileSheetHitTesting(page) {
  const more = page.getByRole("button", { name: "More", exact: true });
  await more.click();
  const sheet = page.locator(".mobile-more-sheet");
  const panel = page.locator(".mobile-more-panel");
  await sheet.waitFor({ state: "visible" });
  const result = await page.evaluate(() => {
    const sheet = document.querySelector(".mobile-more-sheet");
    const panel = document.querySelector(".mobile-more-panel");
    const backdrop = document.querySelector(".mobile-more-backdrop");
    if (!sheet || !panel || !backdrop) return null;
    const panelRect = panel.getBoundingClientRect();
    const panelHit = document.elementFromPoint(panelRect.left + (panelRect.width / 2), panelRect.top + Math.min(20, panelRect.height / 2));
    const backdropHit = document.elementFromPoint(window.innerWidth / 2, window.innerHeight - 2);
    const sheetStyle = getComputedStyle(sheet);
    return {
      panelContainsHit: panel.contains(panelHit),
      backdropContainsHit: backdrop.contains(backdropHit),
      sheet: { position: sheetStyle.position, zIndex: sheetStyle.zIndex },
    };
  });
  assert.ok(result, "mobile more sheet renders its overlay nodes");
  assert.equal(result.sheet.position, "fixed");
  assert.equal(result.sheet.zIndex, "70");
  assert.equal(result.panelContainsHit, true, "mobile sheet panel wins hit-testing inside its bounds");
  assert.equal(result.backdropContainsHit, true, "mobile sheet backdrop wins hit-testing above bottom navigation");
  await panel.getByRole("button", { name: "Close more menu" }).click();
  await sheet.waitFor({ state: "detached" });
}

const browser = await chromium.launch({ headless: true });
try {
  // Each theme characterizes every workspace at desktop, where all workspace
  // interiors are available. Each theme also crosses the shell through every
  // responsive breakpoint, including the mobile overlay/hit-test path.
  for (const theme of ["light", "dark"]) {
    const context = await browser.newContext({ viewport: VIEWPORTS.desktop, colorScheme: theme });
    await context.addInitScript((nextTheme) => localStorage.setItem("survng.theme", nextTheme), theme);
    // Keep this browser-only campaign independent from the server's active
    // authentication configuration. All other API calls remain real, which
    // intentionally exercises each workspace's empty/error handling.
    await context.route("**/api/auth/session", (route) => route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ enabled: false, bootstrap_required: false, user: null }),
    }));
    const page = await context.newPage();
    for (const workspace of WORKSPACES) {
      await openWorkspace(page, workspace);
      assert.equal(await page.locator("html").getAttribute("data-theme"), theme);
      await characterizeCascade(page, workspace, theme, "desktop");
      await assertShellStacking(page, "desktop");
    }
    for (const [viewportName, viewport] of Object.entries(VIEWPORTS)) {
      await page.setViewportSize(viewport);
      await openWorkspace(page, WORKSPACES[0]);
      await characterizeCascade(page, WORKSPACES[0], theme, viewportName);
      await assertShellStacking(page, viewportName);
      if (viewportName === "mobile") await assertMobileSheetHitTesting(page);
    }
    await context.close();
  }
} finally {
  await browser.close();
}

console.log("cascade characterization tests passed");
