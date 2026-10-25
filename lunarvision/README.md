# LunarVision — SIH26166 Prototype
Team THE UNSOLVED — CGC University, Mohali

Multi-modal, Sun-angle and scale invariant image correspondence for
Chandrayaan-2 OHRC / TMC-2 / IIRS imagery.

---

## What this is

A working local prototype (not a mockup) that:
- Detects feature correspondences between two lunar images using SIFT or ORB
- Rejects outliers with RANSAC and estimates a homography
- Registers (warps) the source image into the reference frame
- Reports honest, measured metrics: total matches, inliers, inlier ratio, RMSE
- Lets you generate **synthetic** test pairs with a known ground-truth transform
  (since real Chandrayaan-2 data may not be available yet) — clearly labeled
  as synthetic everywhere in the UI
- Saves observations to a local SQLite database
- Combines multiple saved observations (that share a common reference image)
  into a **regional lunar mosaic**

**Honesty rule baked into the app:** synthetic results are never presented as
real ISRO/Chandrayaan-2 results. Every observation is tagged REAL or
SYNTHETIC throughout the UI and database.

---

## Folder structure

```
lunarvision/
├── backend/           FastAPI + OpenCV backend (the actual CV engine + API)
│   ├── app/
│   │   ├── cv_pipeline/     preprocessing, detectors, matcher, RANSAC, metrics, synthetic generator
│   │   ├── mosaic_engine/   regional mosaic stitcher
│   │   ├── routers/         API endpoints (analyze, observations, mosaic)
│   │   ├── main.py          FastAPI app
│   │   ├── config.py
│   │   └── database.py      SQLite persistence
│   ├── data/                raw / synthetic / uploads / processed images
│   ├── requirements.txt
│   └── run.py
└── frontend/           Static HTML/CSS/JS dashboard (no build step needed)
    ├── index.html            Dashboard
    ├── analyze.html          Main analysis workflow
    ├── observations.html     Saved observations browser
    ├── lunar_map.html        Regional mosaic builder
    ├── css/style.css
    └── js/                   api.js, sidebar.js, dashboard.js, analyze.js, observations.js, lunar_map.js
```

---

## 1. Setup (Windows PowerShell)

```powershell
# From the folder containing lunarvision\
cd lunarvision\backend

# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once (as your normal
user, not admin) and try activating again:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS / Linux equivalent
```bash
cd lunarvision/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Run the backend

```powershell
# still inside lunarvision\backend, with venv activated
python run.py
```

You should see `Application startup complete.` The API is now live at
`http://127.0.0.1:8000`. Open `http://127.0.0.1:8000/docs` in a browser to see
the interactive Swagger API docs and test any endpoint directly.

Leave this terminal running.

---

## 3. Run the frontend

Open a **second** PowerShell terminal:

```powershell
cd lunarvision\frontend
python -m http.server 5500
```

Now open your browser to:

```
http://127.0.0.1:5500/index.html
```

That's the full dashboard. The frontend talks to the backend at
`http://127.0.0.1:8000` (configured in `frontend/js/api.js` — change
`API_BASE` there if you run the backend on a different host/port).

---

## 4. Try the demo flow

1. Open the **Dashboard** — empty at first.
2. Go to **Analyze**. Keep "Use synthetic test pair" checked, adjust
   rotation/scale/translation if you like, click **Generate Synthetic Pair**.
3. Choose sensors (e.g. Source = TMC-2, Reference = OHRC) and a detector
   (SIFT recommended first), click **Analyze**.
4. Watch the pipeline stage indicator, then review:
   - Total Matches / Inliers / Inlier Ratio / RMSE
   - Ground-truth corner error (only shown for synthetic pairs — this is the
     strongest honesty check available since we know the true transform)
   - Correspondence visualization (green = inlier, red = outlier)
   - Before/after registration slider
   - Red/cyan alignment overlay
5. Click **Save Observation**.
6. Repeat step 2–5 once or twice more **using the same reference image**
   (uncheck synthetic and upload the same reference file twice with different
   source images, or regenerate synthetic pairs and note only mosaic pairs
   sharing one reference will align meaningfully).
7. Go to **Lunar Map**, select the saved observations that share a reference
   image, click **Build Mosaic** → see the regional composite.
8. Go to **Saved Observations** to browse everything you've saved.

---

## 5. Using real Chandrayaan-2 data later

Real OHRC/TMC-2/IIRS data can be requested from ISRO's PRADAN portal
(https://pradan.issdc.gov.in/pradan/). Once you have image files (converted
to PNG/JPG/TIFF as needed), just upload them directly on the **Analyze**
page with "Use synthetic test pair" unchecked — the exact same pipeline runs
on real images. The UI will correctly tag those observations as **REAL**
instead of **SYNTHETIC**, and no ground-truth-error metric will show (since
we have no independent ground truth for real observations — only the
RANSAC/self-consistency RMSE).

---

## 6. Known limitations (be upfront about these in your demo)

- SIFT/ORB + RANSAC is a strong classical baseline, not a learned multi-modal
  matcher — genuinely different sensor modalities (e.g. IIRS spectral bands
  vs. OHRC panchromatic) may need additional preprocessing or a learned
  matcher (LoFTR/SuperGlue) to work well; the architecture is modular so this
  can be added as another `detector_name` option later.
- The mosaic blending is simple averaging, not multi-band/Laplacian blending
  — visible seams are expected and are an honest depiction of the current
  method, not a defect being hidden.
- RMSE without ground truth reflects self-consistency of the estimated
  homography with the inlier matches, not absolute geographic accuracy.
