# VerifAura — Document Forgery Detection

live demo - https://verifaura-ai.onrender.com

AI-powered forgery detection using **Error Level Analysis (ELA)** + **Deep CNN**, trained on the CASIA v2.0 dataset.  
Supports images (PNG, JPG, BMP, TIFF) and documents (PDF, DOCX, PPTX) — documents are analysed page-by-page.

## How it works

1. **ELA preprocessing** (`src/ela.py`) — resave the file at a fixed JPEG quality, diff it against the original, amplify the result. Spliced/edited regions show a different compression error level.
2. **CNN classifier** (`src/model.py`) — trained on ELA images to classify authentic vs. forged.
3. **Document extraction** (`src/document_processor.py`) — PDFs are rasterized page-by-page, DOCX/PPTX images are extracted, then each page goes through the same ELA + CNN pipeline.

---

## Project structure

```
VerifAura-project/
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
│   └── best_model_smoke.keras # Trained model
├── data/                      # Not committed (too large)
└── requirements.txt
```

---

## Local setup

```bash
git clone https://github.com/aashya-madhan/VerifAura-AI.git
cd VerifAura-AI

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
cd src
python app.py
```
Open `http://localhost:5000`

---

## Uploading to GitHub

```bash
git add .
git commit -m "your message"
git push origin main
```

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
