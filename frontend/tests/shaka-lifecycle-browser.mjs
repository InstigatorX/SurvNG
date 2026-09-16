import assert from "node:assert/strict";
import { createServer } from "vite";
import { chromium } from "playwright";
import { fileURLToPath } from "node:url";
const server=await createServer({root:fileURLToPath(new URL("..",import.meta.url)),configFile:false,
 server:{host:"127.0.0.1",port:0},plugins:[{name:"shaka-lifecycle-test",enforce:"pre",
 configureServer(s){s.middlewares.use("/test",(_req,res)=>{res.setHeader("Content-Type","text/html");res.end('<div id="root"></div><script type="module" src="/entry.jsx"></script>');});},
 resolveId(id){if(id==="shaka-player")return "\0"+id;if(id==="/entry.jsx")return id;},
 load(id){if(id==="\0shaka-player")return `
 class Player extends EventTarget {
  static isBrowserSupported(){return true;} configure(){}
  attach(){return new Promise((_resolve,reject)=>{window.rejectOldAttach=reject;});}
  destroy(){return Promise.resolve();}
 }
 export default {Player,polyfill:{installAll(){}}};`;
 if(id==="/entry.jsx")return `import React from 'react';import{createRoot}from'react-dom/client';import{ShakaVideo}from'/src/shared/media.jsx';
 function App(){const[visible,setVisible]=React.useState(true);const[errors,setErrors]=React.useState(0);return <><button onClick={()=>setVisible(false)}>Close</button><output>{errors}</output>{visible?<ShakaVideo src="test.m3u8" onError={()=>setErrors(n=>n+1)}/>:null}</>;}
 createRoot(document.getElementById('root')).render(<App/>);`;
 }}]});
let browser;
try{await server.listen();browser=await chromium.launch({headless:true,executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH});const p=await browser.newPage();p.setDefaultTimeout(15000);
 await p.goto(`http://127.0.0.1:${server.httpServer.address().port}/test`);
 await p.waitForFunction(()=>typeof window.rejectOldAttach==="function");
 await p.getByRole("button",{name:"Close"}).click();
 await p.evaluate(async()=>{window.rejectOldAttach(new Error("attach aborted during destroy"));await new Promise(r=>setTimeout(r,100));});
 assert.equal(await p.locator("output").textContent(),"0","a disposed player must not fail its replacement");
 console.log("Disposed Shaka attach rejection is ignored");
}finally{await browser?.close();await server.close();}
