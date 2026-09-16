import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
const temporary = mkdtempSync(join(tmpdir(), "native-replay-"));
execFileSync("ffmpeg", ["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=10", "-t", "24", "-g", "100", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", join(temporary, "test.mp4")]);
execFileSync("ffmpeg", ["-v", "error", "-i", join(temporary,"test.mp4"), "-c", "copy", "-hls_time", "10", "-hls_segment_type", "fmp4", "-hls_playlist_type", "vod", join(temporary,"test.m3u8")]);
const clip = readFileSync(join(temporary, "test.mp4"));
const epoch = 1789500000;
const event = {id:1, camera_id:"test", created_at:new Date(epoch*1000).toISOString(), objects:[], object_tracking:{implementation:"gvatrack", recording_overlay_compatible:false, sample_fps:5, frame_width:640, frame_height:360, tracks:[{track_id:1, label:"person", box:{x1:50,y1:50,x2:150,y2:250}, box_history:[[epoch,50,50,150,250],[epoch+23,80,50,180,250]], trajectory:[[epoch,100,150],[epoch+23,130,150]]}]}};
const server = await createServer({ root:fileURLToPath(new URL("..",import.meta.url)), configFile:false, server:{host:"127.0.0.1",port:0}, plugins:[{
  name:"native-replay-test", configureServer(server) { server.middlewares.use((req,res,next)=>{
    if(req.url.startsWith("/test?")) {res.setHeader("Content-Type","text/html"); res.end('<div id="root"></div><style>.object-track-video-layer{width:640px;height:360px}.event-detail-media{width:640px;height:360px}</style><script type="module" src="/entry.jsx"></script>');return;}
    if(req.url.startsWith("/api/event-clip/settings")){res.setHeader("Content-Type","application/json");res.end('{"before_seconds":0,"after_seconds":3}');return;}
    if(req.url.includes("stream.m3u8")){res.setHeader("Content-Type","application/vnd.apple.mpegurl");res.end(readFileSync(join(temporary,"test.m3u8"),"utf8").replace("#EXTINF:",`#EXT-X-PROGRAM-DATE-TIME:${new Date(epoch*1000).toISOString()}\n#EXTINF:`));return;}
    const segment=/\/(init\.mp4|test\d+\.m4s)(?:\?|$)/.exec(req.url);
    if(segment){res.setHeader("Content-Type","video/mp4");res.end(readFileSync(join(temporary,segment[1])));return;}
    if(req.url.includes("clip.mp4")){res.setHeader("Content-Type","video/mp4");res.end(clip);return;}
    if(req.url.startsWith("/api/")){res.statusCode=404;res.end();return;}
    next();
  });},
  resolveId(id){if(id==="/entry.jsx")return id;},
  load(id){if(id==="/entry.jsx")return `import React from 'react';import {createRoot} from 'react-dom/client';import {IncidentClipLayer} from '/src/incidents/IncidentCard.jsx';import {EventOverlay} from '/src/shared/evidence.jsx';const event=${JSON.stringify(event)};createRoot(document.getElementById('root')).render(location.search.includes('modal')?<EventOverlay event={event} events={[event]} timeZone="UTC" onClose={()=>{}} onSelect={()=>{}}/>:<IncidentClipLayer event={event} active={true} analysisMode="tracks"/>);`;}
}]});
let browser;
try{
 await server.listen();browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});
 const context=await browser.newContext({hasTouch:true,isMobile:true,viewport:{width:1000,height:800}});
 const page=await context.newPage();const errors=[];page.on("pageerror",e=>errors.push(e.message));
 const base=`http://127.0.0.1:${server.httpServer.address().port}`;
 await page.goto(base+"/test?card");
 await page.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
 assert.match(await page.locator("video").getAttribute("src"),/source=live/);
 await page.locator("video").evaluate(v=>{v.pause();v.currentTime=1;});
 await page.locator(".object-track-video-box").waitFor();
 await page.goto(base+"/test?modal");
 await page.getByRole("button",{name:"Show stored object tracks",exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('a[aria-label="Download event video"]')?.href.includes('source=live'));
 await page.locator(".event-detail-media").click();
 await page.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
 assert.match(await page.locator("video").getAttribute("src"),/source=live/);
 await page.locator("video").evaluate(v=>{v.pause();v.currentTime=1;});
 await page.locator(".object-track-video-box").waitFor();
 const desktop=await browser.newPage({viewport:{width:1000,height:800}});
 desktop.on("pageerror",e=>errors.push(e.message));
 const manifest=desktop.waitForRequest(request=>request.url().includes("stream.m3u8"));
 await desktop.goto(base+"/test?card");
 const request=await manifest;
 assert.equal(Number(new URL(request.url()).searchParams.get("after")),26,"clip covers the full 23-second native track plus padding");
 await desktop.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
 await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=9.5;});
 await desktop.waitForFunction(()=>!document.querySelector("video")?.seeking);
 await desktop.locator("video").evaluate(v=>v.play());
 await desktop.waitForFunction(()=>document.querySelector("video")?.currentTime>=10.5);
 await desktop.locator(".object-track-video-box").waitFor();
 await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=22;});
 await desktop.waitForFunction(()=>{const v=document.querySelector("video");return v&&!v.seeking&&v.currentTime>=22;});
 await desktop.locator(".object-track-video-box").waitFor();
 assert.deepEqual(errors,[]);console.log("Both native replay surfaces render stored boxes over the recorded substream");
}finally{await browser?.close();await server.close();rmSync(temporary,{recursive:true,force:true});}
