import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy API calls to Flask during local development (Flask default port 5000).
const proxyPaths = [
  "health",
  "predict",
  "train",
  "self-train",
  "self_train",
  "metrics",
  "before-after",
  "reload-model",
  "charts",
];

const proxy = {};
for (const p of proxyPaths) {
  proxy[`/${p}`] = {
    target: "http://127.0.0.1:5000",
    changeOrigin: true,
  };
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy,
  },
});
