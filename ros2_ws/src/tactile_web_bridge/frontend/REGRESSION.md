# Programme Web UI Regression

## Automated Browser Regression

Run from `ros2_ws/src/tactile_web_bridge/frontend`:

```bash
npm install
npx playwright install chromium
npm run test:e2e
```

`npm run test:e2e:headed` keeps Chromium visible for local debugging.

## Real Stack Manual Regression

1. Build the frontend and the `tactile_web_bridge` package.
2. Launch `web_console_stack.launch.py` for the lightweight Web stack, or `programme_system.launch.py` for the full integrated stack.
3. Open the Web UI and verify:
   - `/api/bootstrap` returns `200` and `frontend_ready=true`
   - WebSocket reconnect overlay appears on gateway loss and disappears on recovery
   - prompt submission updates chat and confirmation card
   - editing the confirmation card creates a draft intervention badge
   - Vision hover highlights the candidate box and click stages the label override
   - `target_locked=true` stops at `waiting_execute` until `Execute` is clicked
   - `Re-plan` clears local draft state but preserves chat history
   - hardware tactile updates the heatmap and sparkline charts
   - `Tare` and `Clear Tare` return `200` and change the baseline as expected
   - the raw-frame static warning appears only when a non-zero hardware frame stops changing and clears after motion resumes
   - log export contains intervention metadata
   - `E-Stop` opens the placeholder modal and performs no control action
