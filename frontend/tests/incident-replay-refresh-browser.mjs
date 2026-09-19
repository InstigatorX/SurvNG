import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium, webkit } from "playwright";
import { fileURLToPath } from "node:url";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { execFileSync } from "node:child_process";
const temporary = mkdtempSync(join(tmpdir(), "native-replay-"));
execFileSync("ffmpeg", ["-v", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=10", "-t", "24", "-g", "100", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", join(temporary, "test.mp4")]);
execFileSync("ffmpeg", ["-v", "error", "-i", join(temporary,"test.mp4"), "-c", "copy", "-hls_time", "10", "-hls_segment_type", "fmp4", "-hls_playlist_type", "vod", join(temporary,"test.m3u8")]);
if (process.env.NATIVE_REPLAY_PRODUCTION === "1") {
 execFileSync(process.env.PYTHON || "python3", [fileURLToPath(new URL("./fixtures/generate-incident-replay.py", import.meta.url)), temporary]);
}
const useWebkit = process.env.NATIVE_REPLAY_BROWSER === "webkit";
const clip = readFileSync(join(temporary, "test.mp4"));
const epoch = 1789500000;
const event = {id:1, camera_id:"test", created_at:new Date(epoch*1000).toISOString(), objects:[], object_tracking:{implementation:"gvatrack", recording_overlay_compatible:true, sample_fps:5, frame_width:640, frame_height:360, tracks:[{track_id:1, label:"person", box:{x1:50,y1:50,x2:150,y2:250}, box_history:[[epoch,50,50,150,250],[epoch+23,80,50,180,250]], trajectory:[[epoch,100,150],[epoch+23,130,150]]}]}};
const detail={...event,id:"incident-test-1",representative_event_id:1,start_epoch:epoch,last_epoch:epoch+23,start_at:event.created_at,end_at:new Date((epoch+23)*1000).toISOString(),event_count:1,has_objects:true,labels:["person"],snapshot_path:"available",events:[event]};
const summary={...detail,object_tracking:undefined,events:[{...event,object_tracking:undefined}]};
const secondEvent={...event,id:2};
const secondDetail={...detail,id:"incident-test-2",representative_event_id:2,events:[secondEvent]};
const secondSummary={...secondDetail,object_tracking:undefined,events:[{...secondEvent,object_tracking:undefined}]};
let detailRequests=0;
let failDetail=false;
let holdDetails=false;
let detailArrived=null;
const pendingDetails=[];
const server=await createServer({root:fileURLToPath(new URL("..",import.meta.url)),configFile:false,cacheDir:join(temporary,"vite-cache"),optimizeDeps:{include:["shaka-player"]},server:{host:"127.0.0.1",port:0},plugins:[{
name:"incident-page-replay",configureServer(s){s.middlewares.use((req,res,next)=>{
const json=value=>{res.setHeader("Content-Type","application/json");res.end(JSON.stringify(value));};
if(req.url.startsWith("/test")){res.setHeader("Content-Type","text/html");res.end('<div id="root"></div><script type="module" src="/entry.jsx"></script>');return;}
if(req.url.startsWith("/api/incidents/search")){json({items:[summary,secondSummary],total:2});return;}
if(req.url.startsWith("/api/incidents/detail")){detailRequests++;const respond=()=>{if(failDetail){res.statusCode=503;json({detail:"retry"});}else json(req.url.includes("event_ids=2")?secondDetail:detail);};if(holdDetails){pendingDetails.push(respond);detailArrived?.();}else setTimeout(respond,300);return;}
if(req.url.startsWith("/api/event-clip/settings")){json({before_seconds:0,after_seconds:3});return;}
if(req.url==="/api/cameras"){json([{id:"test",name:"Test"}]);return;}
if(req.url==="/api/config"){json({cameras:[{id:"test"}]});return;}
if(req.url.includes("stream.m3u8")){res.setHeader("Content-Type","application/vnd.apple.mpegurl");res.end(readFileSync(join(temporary,"test.m3u8"),"utf8").replace("#EXTINF:",`#EXT-X-PROGRAM-DATE-TIME:${new Date(epoch*1000).toISOString()}\n#EXTINF:`));return;}
const segment=/\/(init\d*\.mp4|test\d+\.m4s)(?:\?|$)/.exec(req.url);
if(segment){res.setHeader("Content-Type","video/mp4");res.end(readFileSync(join(temporary,segment[1])));return;}
if(req.url.includes("clip.mp4")){res.setHeader("Content-Type","video/mp4");res.end(clip);return;}
if(req.url.startsWith("/api/")){json([]);return;}next();});},
resolveId(id){if(id==="/entry.jsx")return id;},load(id){if(id==="/entry.jsx")return `import React from 'react';import{createRoot}from'react-dom/client';import{IncidentsPage}from'/src/incidents/IncidentsPage.jsx';import '/src/styles.css';createRoot(document.getElementById('root')).render(<IncidentsPage timeZone="UTC" onRecordingContextChange={()=>{}}/>);`;}
}]});
let browser;
try{
await server.listen();browser=await (useWebkit?webkit:chromium).launch({headless:true});const page=await browser.newPage({viewport:{width:1600,height:1000}});page.setDefaultTimeout(15000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
await page.addInitScript(()=>{window.EventSource=class extends EventTarget{constructor(){super();window.appStream=this;}close(){}};});
await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/test`);
const tracks=page.getByRole("button",{name:"Tracks",exact:true});
await tracks.waitFor();await page.waitForFunction(()=>Array.from(document.querySelectorAll("button")).some(b=>b.textContent.trim()==="Tracks"&&!b.disabled));
await tracks.click();await page.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
await page.locator("video").evaluate(v=>{v.muted=true;v.currentTime=9;window.originalVideo=v;});
await page.waitForFunction(()=>!document.querySelector("video")?.seeking);
await page.locator("video").evaluate(v=>v.play());
const count=detailRequests;
holdDetails=true;
const pendingDetail=new Promise(resolve=>{detailArrived=resolve;});
await page.evaluate(()=>window.appStream.dispatchEvent(new MessageEvent("incident",{data:JSON.stringify({event_id:1})})));
await Promise.race([pendingDetail,new Promise((_,reject)=>setTimeout(()=>reject(new Error("No detail refresh arrived")),15000))]);
assert.equal(await tracks.getAttribute("aria-pressed"),"true","refresh must retain Tracks while details are pending");
assert.equal(await page.evaluate(()=>document.querySelector("video")===window.originalVideo),true,"refresh must retain the playing video");
holdDetails=false;
pendingDetails.splice(0).forEach(respond=>respond());
await page.waitForTimeout(500);
assert.ok(detailRequests>count,"refresh must fetch new details");
assert.equal(await tracks.getAttribute("aria-pressed"),"true");
assert.equal(await page.evaluate(()=>document.querySelector("video")===window.originalVideo),true);
await page.waitForFunction(()=>document.querySelector("video")?.currentTime>11);
await page.locator(".object-track-video-box").first().waitFor();
failDetail=true;
await page.evaluate(()=>window.appStream.dispatchEvent(new MessageEvent("incident",{data:JSON.stringify({event_id:1})})));
await page.waitForTimeout(1700);
assert.equal(await tracks.getAttribute("aria-pressed"),"true","failed refresh preserves analysis mode");
assert.equal(await page.evaluate(()=>document.querySelector("video")===window.originalVideo),true,"failed refresh preserves playback");
failDetail=false;
await page.evaluate(()=>window.appStream.dispatchEvent(new MessageEvent("connected",{data:"{}"})));
await page.waitForTimeout(1700);
assert.equal(await page.evaluate(()=>document.querySelector("video")===window.originalVideo),true,"reconnect revalidation preserves playback");
await page.waitForFunction(()=>document.querySelector("video")?.currentTime>22);
assert.equal(await tracks.getAttribute("aria-pressed"),"true");
await page.locator("video").waitFor({state:"detached"});
assert.equal(await tracks.getAttribute("aria-pressed"),"true","complete playback retains the selected analysis mode");
await tracks.click();
await page.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
// Every explicit Replay action starts the complete clip, even on the same source.
for(const mode of ["Clean","Tracks","Tracks"]){
 await page.locator("video").evaluate(v=>{v.pause();v.currentTime=12;});
 await page.waitForFunction(()=>!document.querySelector("video")?.seeking);
 await page.getByRole("button",{name:mode,exact:true}).click();
 await page.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
 assert.ok(await page.locator("video").evaluate(v=>v.currentTime)<1,"Replay "+mode+" must start at the beginning");
}
// A queued refresh must use the selection at execution time, not its old closure.
await page.locator("video").evaluate(v=>v.pause());
await page.evaluate(()=>window.appStream.dispatchEvent(new MessageEvent("incident",{data:JSON.stringify({event_id:1})})));
await page.getByRole("button",{name:"Next incident",exact:true}).click();
await page.waitForFunction(()=>Array.from(document.querySelectorAll("button")).some(b=>b.textContent.trim()==="Tracks"&&!b.disabled));
await tracks.click();
await page.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
await page.locator("video").evaluate(v=>{v.muted=true;window.originalVideo=v;return v.play();});
await page.waitForTimeout(1800);
assert.equal(await tracks.getAttribute("aria-pressed"),"true","queued refresh preserves the new selection's Tracks mode");
assert.equal(await page.evaluate(()=>document.querySelector("video")===window.originalVideo),true,"queued refresh preserves the new selection's player");
assert.deepEqual(errors,[]);console.log("Full incident page retains Tracks playback during evidence refresh");
}finally{await browser?.close();await server.close();rmSync(temporary,{recursive:true,force:true});}
