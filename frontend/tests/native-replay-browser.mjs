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
const isClipRequest = (request) => /(?:stream\.m3u8|clip\.mp4)/.test(request.url());
const clip = readFileSync(join(temporary, "test.mp4"));
const epoch = 1789500000;
const event = {id:1, camera_id:"test", created_at:new Date(epoch*1000).toISOString(), objects:[], object_tracking:{implementation:"gvatrack", recording_overlay_compatible:false, sample_fps:5, frame_width:640, frame_height:360, tracks:[{track_id:1, label:"person", box:{x1:50,y1:50,x2:150,y2:250}, box_history:[[epoch,50,50,150,250],[epoch+23,80,50,180,250]], trajectory:[[epoch,100,150],[epoch+23,130,150]]}]}};
const server = await createServer({ root:fileURLToPath(new URL("..",import.meta.url)), configFile:false, cacheDir:join(temporary,"vite-cache"), optimizeDeps:{include:["shaka-player"]}, server:{host:"127.0.0.1",port:0}, plugins:[{
  name:"native-replay-test", configureServer(server) { server.middlewares.use((req,res,next)=>{
    if(req.url.startsWith("/test?")) {res.setHeader("Content-Type","text/html"); res.end('<div id="root"></div><style>.object-track-video-layer{width:640px;height:360px}.event-detail-media{width:640px;height:360px}</style><script type="module" src="/entry.jsx"></script>');return;}
    if(req.url.startsWith("/api/event-clip/settings")){res.setHeader("Content-Type","application/json");res.end('{"before_seconds":0,"after_seconds":3}');return;}
    if(req.url.includes("stream.m3u8")){res.setHeader("Content-Type","application/vnd.apple.mpegurl");res.end(readFileSync(join(temporary,"test.m3u8"),"utf8").replace("#EXTINF:",`#EXT-X-PROGRAM-DATE-TIME:${new Date(epoch*1000).toISOString()}\n#EXTINF:`));return;}
    const segment=/\/(init\d*\.mp4|test\d+\.m4s)(?:\?|$)/.exec(req.url);
    if(segment){res.setHeader("Content-Type","video/mp4");res.end(readFileSync(join(temporary,segment[1])));return;}
    if(req.url.includes("clip.mp4")){
      res.setHeader("Content-Type","video/mp4");res.setHeader("Accept-Ranges","bytes");
      const range=/^bytes=(\d+)-(\d*)$/.exec(req.headers.range || "");
      const start=range?Number(range[1]):0;const end=range&&range[2]?Math.min(clip.length-1,Number(range[2])):clip.length-1;
      if(range){res.statusCode=206;res.setHeader("Content-Range",`bytes ${start}-${end}/${clip.length}`);}
      res.setHeader("Content-Length",end-start+1);res.end(clip.subarray(start,end+1));return;
    }
    if(req.url.startsWith("/api/")){res.statusCode=404;res.end();return;}
    next();
  });},
  resolveId(id){if(id==="/entry.jsx")return id;},
  load(id){if(id==="/entry.jsx")return `import React from 'react';import {createRoot} from 'react-dom/client';import {IncidentClipLayer} from '/src/incidents/IncidentCard.jsx';import {EventOverlay} from '/src/shared/evidence.jsx';const fullEvent=${JSON.stringify(event)};function growingEvent(end){const item=structuredClone(fullEvent);item.object_tracking.state='active';item.object_tracking.tracks[0].box_history[1][0]=${epoch}+end;item.object_tracking.tracks[0].trajectory[1][0]=${epoch}+end;return item;}function App(){const [event,setEvent]=React.useState(location.search.includes('growing')?growingEvent(8):location.search.includes('late')?{...fullEvent,object_tracking:{...fullEvent.object_tracking,tracks:[]}}:fullEvent);const [open,setOpen]=React.useState(true);window.completeTracks=()=>setEvent({...fullEvent,object_tracking:{...fullEvent.object_tracking,state:"complete"}});window.growTracks=()=>setEvent(growingEvent(15));window.confirmGeometry=()=>setEvent({...fullEvent,object_tracking:{...fullEvent.object_tracking,recording_overlay_compatible:true}});window.alignTiming=()=>setEvent(current=>({...current,object_tracking:{...current.object_tracking,recording_alignment:{source:"main",verified:true,offset_seconds:1}}}));window.reopen=()=>setOpen(value=>!value);return open?(location.search.includes('modal')?<EventOverlay event={event} events={[event]} timeZone="UTC" onClose={()=>setOpen(false)} onSelect={()=>{}}/>:<IncidentClipLayer event={{...event,object_tracking:undefined}} trackingEvent={event} active={true} analysisMode="tracks"/>):null;}createRoot(document.getElementById('root')).render(<App/>);`;}
}]});
let browser;
try{
 await server.listen();browser=useWebkit ? await webkit.launch({headless:true}) : await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});
 const context=await browser.newContext({hasTouch:true,isMobile:true,viewport:{width:1000,height:800}});
 const page=await context.newPage();page.setDefaultTimeout(15000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
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
 desktop.setDefaultTimeout(15000);
 desktop.on("pageerror",e=>errors.push(e.message));
 if(useWebkit) await desktop.addInitScript(()=>{
   // Linux WebKit lacks macOS's native HLS capability. Exercise the same
   // incident transport selection while decoding MP4 in real WebKit.
   const original=HTMLMediaElement.prototype.canPlayType;
   HTMLMediaElement.prototype.canPlayType=function(type){return type==="application/vnd.apple.mpegurl"?"probably":original.call(this,type);};
 });
 const manifest=desktop.waitForRequest(isClipRequest);
 await desktop.goto(base+"/test?card");
 const request=await manifest;
 assert.equal(Number(new URL(request.url()).searchParams.get("after")),26,"clip covers the full 23-second native track plus padding");
 await desktop.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
 await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=9.5;});
 await desktop.waitForFunction(()=>!document.querySelector("video")?.seeking);
 await desktop.locator("video").evaluate(v=>{v.muted=true;return v.play();});
 await desktop.waitForFunction(()=>document.querySelector("video")?.currentTime>=10.5);
 await desktop.locator(".object-track-video-box").waitFor();
 await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=22;});
 await desktop.waitForFunction(()=>{const v=document.querySelector("video");return v&&!v.seeking&&v.currentTime>=22;}).catch(async(error)=>{console.error(await desktop.locator("video").evaluate(v=>({src:v.src,time:v.currentTime,duration:v.duration,seeking:v.seeking,error:v.error?.message,buffered:Array.from({length:v.buffered.length},(_,i)=>[v.buffered.start(i),v.buffered.end(i)])})));throw error;});
 await desktop.locator(".object-track-video-box").waitFor();
 if(!useWebkit){
   await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=12;});
   await desktop.waitForFunction(()=>!document.querySelector("video")?.seeking);
   await desktop.locator("video").evaluate(v=>v.dispatchEvent(new Event("error")));
   await desktop.waitForFunction(()=>{const v=document.querySelector("video");return v?.src.includes("clip.mp4")&&v.readyState>=2&&v.currentTime>=11.9;});
   assert.ok(await desktop.locator("video").evaluate(v=>v.currentTime)<14,"MP4 fallback preserves the playback position");
 }
 for (const surface of ["card", "modal"]) {
   await desktop.goto(base+"/test?"+surface+"&late");
   await desktop.waitForFunction(()=>typeof window.completeTracks === "function");
   await desktop.waitForTimeout(300);
   const fullManifest=surface === "card" ? desktop.waitForRequest(r=>isClipRequest(r)&&Number(new URL(r.url()).searchParams.get("after"))===26) : null;
   await desktop.evaluate(()=>window.completeTracks());
   if(fullManifest) await fullManifest;
   else await desktop.waitForFunction(()=>Number(new URL(document.querySelector('a[aria-label="Download event video"]')?.href || location.href).searchParams.get("after"))===26);
   for(let repeat=0;repeat<3;repeat++) {
     if(surface==="modal") {
       await desktop.getByRole("button",{name:"Show stored object tracks",exact:true}).click();
       await desktop.waitForFunction(()=>document.querySelector('a[aria-label="Download event video"]')?.href.includes('source=live'));
       await desktop.locator(".event-detail-media").click();
     }
     await desktop.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
     await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=22;});
     await desktop.waitForFunction(()=>!document.querySelector("video")?.seeking);
     await desktop.locator(".object-track-video-box").waitFor();
     await desktop.evaluate(()=>window.reopen());
     await desktop.locator("video").waitFor({state:"detached"});
     if(repeat<2) await desktop.evaluate(()=>window.reopen());
   }
 }
 for(const surface of ["card","modal"]){
   await desktop.goto(base+"/test?"+surface+"&growing");
   if(surface==="modal"){
     await desktop.getByRole("button",{name:"Show stored object tracks",exact:true}).click();
     await desktop.waitForFunction(()=>document.querySelector('a[aria-label="Download event video"]')?.href.includes('source=live'));
     await desktop.locator(".event-detail-media").click();
   }
   await desktop.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
   await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=4;window.originalVideo=v;});
   await desktop.waitForFunction(()=>!document.querySelector("video")?.seeking);
   await desktop.evaluate(()=>window.growTracks());
   await desktop.waitForTimeout(150);
   assert.equal(await desktop.evaluate(()=>document.querySelector("video")===window.originalVideo),true,"active updates must not replace the playing window");
   assert.ok(Math.abs(await desktop.locator("video").evaluate(v=>v.currentTime)-4)<.25);
   await desktop.evaluate(()=>window.completeTracks());
   await desktop.waitForFunction(()=>{const v=document.querySelector("video");return v&&v!==window.originalVideo&&v.readyState>=2&&v.currentTime>=3.9;});
   await desktop.locator("video").evaluate(v=>v.pause());
   assert.ok(await desktop.locator("video").evaluate(v=>v.currentTime)<6,"completion must resume near the old position");
 }
 for(const surface of ["card","modal"]){
   await desktop.goto(base+"/test?"+surface);
   await desktop.waitForFunction(()=>typeof window.confirmGeometry === "function");
   await desktop.evaluate(()=>window.confirmGeometry());
   if(surface==="modal"){
     await desktop.getByRole("button",{name:"Show stored object tracks",exact:true}).click();
     await desktop.waitForFunction(()=>document.querySelector('a[aria-label="Download event video"]')?.href.includes('source=main'));
     await desktop.locator(".event-detail-media").click();
   }
   await desktop.waitForFunction(()=>document.querySelector("video")?.readyState>=2);
   await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=22;});
   await desktop.waitForFunction(()=>{const v=document.querySelector("video");return v&&!v.seeking&&v.currentTime>=22;});
   await desktop.locator(".object-track-video-box").waitFor();
   await desktop.locator("video").evaluate(v=>{v.pause();v.currentTime=4;window.originalVideo=v;});
   await desktop.waitForFunction(()=>!document.querySelector("video")?.seeking);
   await desktop.evaluate(()=>window.alignTiming());
   await desktop.waitForFunction(()=>Math.abs(Number(document.querySelector(".object-track-video-box")?.getAttribute("x"))-(50+30*5/23))<.2);
   assert.equal(await desktop.evaluate(()=>document.querySelector("video")===window.originalVideo),true,"timing correction must not replace the video");
   assert.ok(Math.abs(await desktop.locator("video").evaluate(v=>v.currentTime)-4)<.1,"timing correction must not seek the video");
 }
 assert.deepEqual(errors,[]);console.log("Both replay surfaces retain tracks across segments and use confirmed main-stream geometry");
}finally{await browser?.close();await server.close();rmSync(temporary,{recursive:true,force:true});}
