import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: proxy API calls to the uvicorn backend so the frontend can use same-origin /api.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
