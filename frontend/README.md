# Frontend (not yet built)

Per the MVP scope's build order (Section 11), the frontend is a Week 6-7
deliverable, built once the backend API (Section 7) is working against a
real trained checkpoint. This folder is a placeholder for that phase.

Planned shape (Section 9): a single-page app with three UI states, not a
multi-route dashboard.

```
frontend/
  src/
    components/   # Uploader, ResultsView, Heatmap
    pages/         # single page app
    services/      # API client
```

- Stack: React + TypeScript + Vite + Tailwind
- No React Query / Zustand — local component state is enough for 3 screens
- No patients module, no analytics, no admin panel (Section 2)

States: **Upload** (file picker, calls `/api/eeg/upload` then
`/api/eeg/{id}/predict`) → **Processing** (spinner while the synchronous
request is in flight) → **Results** (three risk score cards, the Grad-CAM
heatmap image, the plain-language report text).
