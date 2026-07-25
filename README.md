# VerifAura AI — Document Forgery Detection

AI-powered forgery detection using **Error Level Analysis (ELA)** + **Deep CNN**, trained on the CASIA v2.0 dataset.  
Supports images (PNG, JPG, BMP, TIFF) and documents (PDF, DOCX, PPTX) — documents are analysed page-by-page.

## How it works

1. **ELA preprocessing** (`src/ela.py`) — resave the file at a fixed JPEG quality, diff it against the original, amplify the result. Spliced/edited regions show a different compression error level.
2. **CNN classifier** (`src/model.py`) — trained on ELA images to classify authentic vs. forged.
3. **Document extraction** (`src/document_processor.py`) — PDFs are rasterized page-by-page, DOCX/PPTX images are extracted, then each page goes through the same ELA + CNN pipeline.

---

## Project structure

```
VerifAura-AI_project/
├── src/
│   ├── app.py                 # Flask web app
│   ├── predict.py             # predict_image() + predict_document()
│   ├── ela.py                 # ELA algorithm
│   ├── document_processor.py  # PDF/DOCX/PPTX → images
│   ├── model.py               # CNN architecture
│   ├── train.py               # Training loop
│   └── prepare_dataset.py     # Build ELA dataset from raw images
├── templates/
│   ├── index.html             # Upload page
│   └── result.html            # Results page
├── static/
│   ├── style.css
│   └── uploads/               # Runtime upload storage
├── models/
│   └── best_model_smoke.keras # Trained model (committed to repo)
├── data/                      # Not committed (too large)
├── Procfile                   # Render / Railway / Heroku
├── render.yaml                # Render.com deploy config
├── runtime.txt                # Python version
└── requirements.txt
```

---

## Local setup

```bash
git clone https://github.com/YOUR_USERNAME/VerifAura-AI_project.git
cd VerifAura-AI_project

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Run the app:
```bash
cd src
python app.py
```
Open `http://localhost:5000`

---

## Uploading to GitHub

### First time

```bash
# Inside the project folder
git init                          # skip if already initialised
git add .
git commit -m "initial commit"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/VerifAura-AI_project.git
git branch -M main
git push -u origin main
```

### After making changes

```bash
git add .
git commit -m "your message"
git push
```

> **Note:** `data/raw/`, `data/processed/`, `venv/`, and the large model files are all in `.gitignore` — they won't be pushed.  
> `models/best_model_smoke.keras` (0.37 MB) **is** committed and included so the app works after deploy.

---

## Deployment

### Option 1 — Render.com (recommended, free tier available)

1. Push your code to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repo.
4. Render auto-detects `render.yaml`. Click **Deploy**.
5. First deploy takes ~5 min (TensorFlow install). Your app will be live at `https://verifaura-ai.onrender.com`.

The `render.yaml` already configures:
- Python runtime
- Install command
- Start command (`gunicorn`)
- A persistent disk for uploaded files

> Free tier spins down after 15 min of inactivity. First request after spin-down takes ~30 sec.

---

### Option 2 — Railway.app (free tier, faster cold starts)

1. Push to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**.
3. Select your repo. Railway picks up the `Procfile` automatically.
4. Add one environment variable: `PORT` → `8080` (Railway sets this automatically, but confirm it's present).
5. Click **Deploy**. Done.

---

### Option 3 — Heroku

```bash
# Install Heroku CLI, then:
heroku create verifaura-ai
git push heroku main
heroku open
```

Heroku uses the `Procfile`. Note: Heroku's free tier was discontinued — you need a paid plan.

---

### Option 4 — Docker (any VPS / cloud provider)

Create a `Dockerfile` in the project root:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p static/uploads

EXPOSE 8080
CMD ["gunicorn", "--chdir", "src", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120"]
```

Build and run:
```bash
docker build -t verifaura-ai .
docker run -p 8080:8080 verifaura-ai
```

Deploy to any provider that runs Docker: DigitalOcean App Platform, Google Cloud Run, AWS App Runner, Fly.io.

---

## Training your own model

1. Get the CASIA v2.0 dataset. Place the `Au/` and `Tp/` folders in `data/raw/`.
2. Build the ELA dataset:
   ```bash
   cd src
   python prepare_dataset.py   # set SMOKE_TEST = False for full run
   ```
3. Train:
   ```bash
   python train.py             # set SMOKE_TEST = False for full run
   ```
   Output: `models/best_model.keras`
4. Update `MODEL_PATH` in `src/predict.py` to point to `best_model.keras` for the full model.

---

## Supported file types

| Type | Format | How it's analysed |
|------|--------|-------------------|
| Image | PNG, JPG, JPEG, BMP, TIFF | Single ELA + CNN pass |
| PDF | .pdf | Each page rasterized at 150 DPI, then ELA + CNN |
| Word | .docx, .doc | Embedded images extracted, text rendered if none |
| PowerPoint | .pptx, .ppt | Per-slide images extracted, slide thumbnail if none |

---

## Tech stack

TensorFlow · Flask · Gunicorn · PyMuPDF · python-docx · python-pptx · Pillow · CASIA v2.0
