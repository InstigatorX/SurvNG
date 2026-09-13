import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  base: "/static/",
  experimental: {
    // Resolve lazy chunks and CSS beside their importing JS file. The server
    // rewrites HTML asset URLs for its runtime proxy prefix; an absolute build
    // base in JS would escape that prefix and hit the proxy's root application.
    renderBuiltUrl(_filename, { hostType }) {
      if (hostType === "js") return { relative: true };
    },
  },
  plugins: [react()],
  build: {
    outDir: "../survng/static",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(rootDir, "index.html"),
        recordings: resolve(rootDir, "recordings.html"),
        config: resolve(rootDir, "config.html"),
        onvif: resolve(rootDir, "onvif.html"),
      },
      output: {
        entryFileNames: "assets/[name]-[hash].js",
        chunkFileNames: "assets/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
