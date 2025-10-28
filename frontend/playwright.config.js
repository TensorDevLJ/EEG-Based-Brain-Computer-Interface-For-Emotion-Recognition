const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: 'tests',
  timeout: 60 * 1000,
  use: {
    baseURL: 'http://127.0.0.1:8080',
    headless: true,
    viewport: { width: 1280, height: 720 },
  },
  webServer: {
    // Start both backend and frontend via the root npm script
    command: 'npm run dev',
    port: 8080,
    cwd: '..',
    reuseExistingServer: false,
  },
});
