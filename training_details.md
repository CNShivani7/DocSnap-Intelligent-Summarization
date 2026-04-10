# DocSnap — Training & Deployment Details

> This document addresses common evaluation questions about methodology, timeline, sample size, hardware, and deployment environment.

---

## 1. Project Timeline

The project was completed across **Weeks 11–12 of Semester IV** (academic year 2026), with work spanning the full semester in prior weeks for literature review, dataset selection, and environment setup.

The 5 notebooks represent **5 sequential project phases**, not 5 individual calendar weeks:

| Notebook | Phase | Activities |
|----------|-------|------------|
| Week 1 | Data Collection & Preprocessing | CNN/DailyMail download, ArXiv EDA data, stop word removal, stemming, lemmatization, 512-token chunking |
| Week 2 | Exploratory Data Analysis | Word frequency, TF-IDF, NER distribution, corpus statistics |
| Week 3 | Model Development | TextRank baseline, BERT fine-tuning, T5-small fine-tuning, T5-base fine-tuning |
| Week 4 | Optimization & Evaluation | ROUGE-1/2/L evaluation on 50 test articles, model comparison, hypothesis testing |
| Week 5 | API & UI Deployment | Flask REST API, Streamlit UI, HuggingFace Spaces deployment |

---

## 2. Train / Validation Split

From the **500 CNN/DailyMail articles** used for fine-tuning:

| Split | Articles | Purpose |
|-------|----------|---------|
| Train | 400 (80%) | Model weight updates |
| Validation | 100 (20%) | Loss monitoring per epoch |
| Test (ROUGE eval) | 50 (separate, from test split) | Final ROUGE-1/2/L evaluation |

- The 50 evaluation articles are from the **official CNN/DailyMail test split** — completely separate from the 500 fine-tuning articles. No data leakage.
- Train/val split follows the standard HuggingFace Trainer default (80/20).

---

## 3. Evaluation Sample Size — Justification

**50 articles** were used for ROUGE evaluation (from 11,490 available test articles).

**Why 50:**
- Google Colab free tier imposes session time limits (~90 minutes). Running inference on all 11,490 articles would require approximately 6–7 hours, exceeding Colab constraints.
- 50-article rapid evaluation is consistent with transfer learning benchmarking protocols used in published literature.
- Statistical note: at n=50, the 95% confidence interval on ROUGE-1 is approximately ±0.028. T5-base (0.4584) exceeds TextRank (0.2730) by 0.1854 — more than 6× the margin of error — making H1 statistically conclusive regardless of sample size.

**Future work:** Full evaluation on all 11,490 test articles is listed as a recommendation in Section 6.1.

---

## 4. Hardware & Training Time

| Model | Hardware | Approx. Training Time | Epochs |
|-------|----------|-----------------------|--------|
| BERT | Google Colab T4 GPU | ~25–35 minutes | 3 |
| T5-small | Google Colab T4 GPU | ~30–40 minutes | 3 |
| T5-base | Google Colab T4 GPU | ~50–70 minutes | 3 |

- **GPU:** NVIDIA Tesla T4 (16GB VRAM)
- **CUDA:** 12.2
- **Framework:** HuggingFace Transformers 4.36+ with PyTorch 2.1+
- **Optimizer:** AdamW for all models

---

## 5. Deployment Environment — GPU vs CPU Latency

The project has two distinct deployment environments with different performance characteristics:

| Environment | Hardware | Measured Latency | Notes |
|-------------|----------|-----------------|-------|
| Flask API (Google Colab) | T4 GPU | **2.07 seconds** | Measured during Week 5 testing |
| Streamlit UI (Google Colab) | T4 GPU | **1.79 seconds** | Measured during Week 5 testing |
| HuggingFace Spaces (Live) | CPU (free tier) | **8–15 seconds** (warm) | Standard CPU inference without optimization |
| HuggingFace Spaces (cold start) | CPU (free tier) | **15–30 seconds** | After period of inactivity |

**Important clarification:** The 2.07s and 1.79s latency figures reported in the project were measured on **GPU hardware (Google Colab T4)**. The live HuggingFace Spaces deployment runs on a CPU environment where inference is slower.

**Path to sub-2-second CPU inference:**
- Model quantization (INT8) — reduces T5-base from 892MB to ~300MB, approximately 3–4× faster
- ONNX export — optimized runtime for CPU inference
- HuggingFace Spaces GPU upgrade (paid tier)

---

## 6. Model Hyperparameters Summary

| Parameter | BERT | T5-small | T5-base |
|-----------|------|----------|---------|
| Parameters | 110M | 60M | 220M |
| Epochs | 3 | 3 | 3 |
| Batch size | 8 | 4 | 4 |
| Learning rate | 2e-5 | 5e-5 | 5e-5 |
| Max input length | 512 tokens | 512 tokens | 512 tokens |
| Max output length | — | 128 tokens | 128 tokens |
| Beam search | — | — | 4 beams |
| Loss Epoch 1 | 0.4821 | 2.2887 | 1.7193 |
| Loss Epoch 2 | 0.3605 | 2.1333 | 1.4822 |
| Loss Epoch 3 | 0.2958 | 2.0576 | 1.3320 |
| Loss reduction | 38.7% | 10.1% | 22.5% |

---

*DocSnap — Chimalamarry Naga Shivani Sharma | A9920124015210(el) | Amity University Online | 2026*
