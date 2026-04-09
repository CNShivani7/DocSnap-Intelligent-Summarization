# DocSnap: Intelligent Summarization for Fast Decision-Making

**Amity University Online** | MBA (Data Science) | **Keystone Advisory Group, Chennai** | 2025–2026

---

## Project Overview

DocSnap is an AI-powered document summarization system using **facebook/bart-large-cnn** via Hugging Face Transformers. It summarizes professional documents across financial, healthcare, and news domains.

**ROUGE Results:** ROUGE-1: 0.4210 | ROUGE-2: 0.1985 | ROUGE-L: 0.2934

---

## Output Verification Links

Click any link to verify the output directly — no login required.

| Figure | Description | Link |
|--------|-------------|------|
| **Figure 4.1** | GPU Environment Setup — Tesla T4, 15.78 GB, CUDA 12.2 | [View Output](https://nbviewer.org/gist/CNShivani7/4482d7f02bff16fb384193433d7e740f) |
| **Figure 4.2** | Library Installation — transformers 4.36.0, rouge-score 0.1.2 | [View Output](https://nbviewer.org/gist/CNShivani7/4482d7f02bff16fb384193433d7e740f) |
| **Figure 4.3** | BART Model Loading — facebook/bart-large-cnn, 1.63 GB | [View Output](https://nbviewer.org/gist/CNShivani7/4482d7f02bff16fb384193433d7e740f) |
| **Figure 4.4/4.5** | Streamlit App Output — 83.7% compression, 4.2 sec | [View Output](https://nbviewer.org/gist/CNShivani7/4482d7f02bff16fb384193433d7e740f) |
| **Figure 4.6** | ROUGE Evaluation — rouge1: 0.4210, rouge2: 0.1985, rougeL: 0.2934 | [View Output](https://nbviewer.org/gist/CNShivani7/942c67f2f40b25e161ee8c5a77b977cf) |
| **All Figures** | Full notebook — all 5 outputs in one place | [View All](https://nbviewer.org/gist/CNShivani7/4482d7f02bff16fb384193433d7e740f) |

---

## Model & Dataset

- **Model:** [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- **Dataset:** [CNN/DailyMail 3.0.0](https://huggingface.co/datasets/cnn_dailymail)
- **Evaluation Library:** [rouge-score 0.1.2](https://pypi.org/project/rouge-score/)
- **Paper:** [Lewis et al. (2020) — BART](https://arxiv.org/abs/1910.13461)

---

## Files

| File | Description |
|------|-------------|
| `DocSnap_Colab_PreRun.ipynb` | Full implementation notebook — all cells pre-run |
| `app.py` | Streamlit application source code |
| `Fig4_1_GPU_Setup.ipynb` | Figure 4.1 — GPU environment output |
| `Fig4_2_Library_Install.ipynb` | Figure 4.2 — Library installation output |
| `Fig4_3_BART_Model_Load.ipynb` | Figure 4.3 — BART model loading output |
| `Fig4_4_5_Streamlit_App.ipynb` | Figure 4.4/4.5 — Streamlit app output |
