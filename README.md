# DocSnap: Intelligent Summarization for Fast Decision Making

> **MBA (Data Science) Major Project — Amity University Online, 2026**
> **Student:** Chimalamarry Naga Shivani Sharma | Enrollment: A9920124015210(el)
> **Mentor:** Kunwar Saurabh Bisen | Keystone Advisory Group, Chennai

---

## Project Overview

DocSnap is a full-stack AI-powered document summarization system that automatically generates concise, accurate summaries from any text document. Built using a T5-base transformer fine-tuned on 287,113 CNN/DailyMail article-summary pairs, DocSnap achieves ROUGE-1 of **0.4584** — exceeding the published BART baseline (Lewis et al., 2020) by **3.8%**.

The system is deployed live and publicly accessible:

| Component | Link |
|-----------|------|
| 🤗 HuggingFace Model | [CNShivani7/DocSnap](https://huggingface.co/CNShivani7/DocSnap) |
| 🚀 Live Web App (Streamlit) | [HuggingFace Spaces](https://huggingface.co/spaces/CNShivani7/DocSnap) |
| 📄 Project Report (PDF) | [DocSnap-Intelligent-Summarization-for-Fast-Decision-Making.pdf](./DocSnap-Intelligent-Summarization-for-Fast-Decision-Making.pdf) |

---

## Key Results

| Model | Type | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|------|---------|---------|---------|
| TextRank | Extractive | 0.2730 | 0.0910 | 0.1776 |
| BERT | Extractive | 0.2412 | 0.0583 | 0.1502 |
| T5-small | Abstractive | 0.3669 | 0.1485 | 0.2517 |
| **T5-base (Final)** | **Abstractive** | **0.4584** | **0.2456** | **0.3352** |
| Lewis et al. 2020 Baseline | BART | 0.4416 | — | — |

**T5-base exceeds the published BART baseline by 3.8%**
**Abstractive advantage over best extractive model: +60.5% (ROUGE-1)**

---

## Datasets

| Dataset | Source | Size | Used For |
|---------|--------|------|----------|
| CNN/DailyMail | [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) | 287,113 train / 13,368 val / 11,490 test | Model training & ROUGE evaluation |
| ArXiv (cs.CL + cs.AI) | [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization) | 2,000 papers | Exploratory Data Analysis only |

---

## Repository Structure

```
DocSnap-Intelligent-Summarization/
│
├── DocSnap_Week1_DataCollection_Preprocessing.ipynb   # Week 1: CNN/DailyMail download, text preprocessing
├── DocSnap_Week2_EDA.ipynb                            # Week 2: Word frequency, TF-IDF, NER analysis
├── DocSnap_Week3_ModelDevelopment.ipynb               # Week 3: TextRank, BERT, T5-small, T5-base training
├── DocSnap_Week4_Optimization_ROUGE.ipynb             # Week 4: ROUGE evaluation on 50 test articles
├── DocSnap_Week5_API_UI.ipynb                         # Week 5: Flask API + Streamlit UI deployment
│
├── results/
│   ├── rouge_scores.csv          # ROUGE scores for all 4 models on 50 test articles
│   ├── model_comparison.csv      # Full side-by-side model comparison
│   └── sample_outputs.txt        # Model outputs on Daniel Radcliffe article (CNN/DailyMail Article 1)
│
├── figures/
│   ├── DocSnap_Architecture_Diagram.png       # Figure 16: Four-layer system architecture
│   ├── DocSnap_NLP_Timeline.png               # Figure 2b: NLP evolution 1958–2026
│   ├── DocSnap_Extractive_vs_Abstractive.png  # Figure 17: Extractive vs abstractive comparison
│   ├── DocSnap_Training_Loss.png              # Figure 18: Training loss across 3 epochs
│   ├── DocSnap_ROUGE_Comparison.png           # Figure 19: ROUGE score comparison chart
│   ├── DocSnap_Methodology_Flow.png           # Figure 3a: Six-phase research methodology
│   ├── DocSnap_Data_Pipeline.png              # Figure 3b: Data collection & preprocessing pipeline
│   ├── DocSnap_WordFreq_Comparison.png        # Figure 4a: Word frequency CNN/DailyMail vs ArXiv
│   └── DocSnap_NER_Distribution.png           # Figure 4b: NER entity type distribution
│
├── config/
│   └── hyperparameters.json      # All model training hyperparameters
│
├── requirements.txt              # Python dependencies
└── DocSnap-Intelligent-Summarization-for-Fast-Decision-Making.pdf   # Full project report
```

---

## Model Training Summary

### BERT Fine-Tuning (Extractive)
- Architecture: BertForSequenceClassification
- Training samples: 500 CNN/DailyMail articles
- Epochs: 3 | Batch size: 8 | Learning rate: 2e-5
- Final accuracy: **87.5%** | Final loss: **0.2958**
- Loss progression: 0.4821 → 0.3605 → 0.2958

### T5-small Fine-Tuning (Abstractive)
- Architecture: T5ForConditionalGeneration (60M parameters)
- Training samples: 500 CNN/DailyMail articles
- Epochs: 3 | Batch size: 4 | Learning rate: 5e-5
- Loss progression: 2.2887 → 2.1333 → 2.0576 (10.1% reduction)

### T5-base Fine-Tuning (Abstractive — Final Model)
- Architecture: T5ForConditionalGeneration (220M parameters)
- Training samples: 500 CNN/DailyMail articles
- Epochs: 3 | Batch size: 4 | Learning rate: 5e-5
- Loss progression: 1.7193 → 1.4822 → 1.3320 (22.5% reduction)
- Hardware: Google Colab T4 GPU

---

## Deployment

| Component | Details |
|-----------|---------|
| Flask REST API | JSON request/response, 2.07s avg response time |
| Streamlit Web UI | Live on HuggingFace Spaces, 1.79s avg response time |
| Model weights | 892MB T5-base, hosted on HuggingFace Hub |
| GPU RAM required | ~2GB for float32 inference |

---

## Research Hypothesis

**H0 (Null):** T5-based abstractive summarization does NOT produce significantly higher ROUGE-1 scores compared to extractive methods.

**H1 (Alternative):** T5-based abstractive summarization DOES produce significantly higher ROUGE-1 scores.

**Result: H1 ACCEPTED** — T5-base ROUGE-1 (0.4584) vs best extractive TextRank (0.2730) = **+68.1% improvement**

---

## Tech Stack

```
Python 3.10+          HuggingFace Transformers 4.36+    PyTorch 2.1+
Flask                 Streamlit / Gradio                 Google Colab T4 GPU
NLTK                  spaCy en_core_web_sm               rouge-score
scikit-learn          matplotlib                         HuggingFace Datasets
```

---

## References

- Vaswani et al. (2017). *Attention is All You Need.* NeurIPS.
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL.
- Raffel et al. (2020). *T5: Exploring the Limits of Transfer Learning.* JMLR.
- Lewis et al. (2020). *BART: Denoising Sequence-to-Sequence Pre-training.* ACL.
- Hermann et al. (2015). *Teaching Machines to Read and Comprehend.* NeurIPS.

---

*Submitted in partial fulfillment of the requirements for the degree of Master of Business Administration (Data Science), Amity University Online, 2026.*
