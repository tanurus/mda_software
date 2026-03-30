# Detrital Zircon MDA Web App

This repository contains a **web application** (built with Streamlit) for calculating **Maximum Depositional Age (MDA)** metrics from detrital zircon data.

If your question is: **"What is this, and what should I do with it?"**
- This is the **source code** for your calculator.
- To make it available online, you should **deploy it** to a hosting service.
- The easiest option is **Streamlit Community Cloud** (free tier available).

---

## What this app does

- Calculates 10 MDA methods:
  - YSG, YDZ, YC1σ, YC2σ, Y3Za, Y3Zo_2σ, YSP, MLA, YPP, τ
- Accepts multiple samples in one paste using alternating columns:
  - `sample1_age, sample1_sigma, sample2_age, sample2_sigma, ...`
- Supports input uncertainty formats:
  - 1σ absolute, 2σ absolute, 1σ percent, 2σ percent
- Supports output uncertainty formats:
  - 1σ absolute, 2σ absolute, 1σ percent, 2σ percent
- Produces method tables and interactive plots.

---

## What you should do next (recommended)

### Option A — Deploy to Streamlit Community Cloud (fastest)

1. Create a GitHub repository and push this code.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Sign in and click **New app**.
4. Select your repo, branch, and set:
   - **Main file path**: `app.py`
5. Click **Deploy**.
6. Share the generated URL.

This gives you a public web app that users can open in a browser (no local install needed by them).

### Option B — Deploy on your own server (more control)

Use a VM/container service (e.g., AWS, Azure, GCP, DigitalOcean), then run:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Then expose the app behind HTTPS using a reverse proxy (Nginx/Caddy) and a domain name.

---

## Run locally (for testing before deployment)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in terminal (usually `http://localhost:8501`).

---

## Data format example

```csv
sample1_age,sample1_sigma,sample2_age,sample2_sigma
518.37,11.03,610.1,8.2
525.66,11.10,603.4,7.9
529.80,11.44,599.7,8.1
535.77,11.16,590.1,7.8
```

---

## Important scientific notes

- YDZ returns Monte Carlo-based minimum-age summary (`median`, `p2.5`, `p97.5`).
- YPP returns the youngest KDE peak age.
- MLA in this code is a **Gaussian-mixture/BIC approximation** of IsoplotR-style youngest-component unmixing.
- Ensure your zircon dataset is pre-filtered as needed (e.g., discordancy filter) before input.
