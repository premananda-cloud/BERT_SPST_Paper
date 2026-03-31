# BERT + SPST Fake News Detection — Report & Results

> **97.52% accuracy · 97.71% F1 · 0.9981 AUC**  
> Resource-efficient BERT fine-tuning via Sequential Parameter Segment Training (SPST)

This repository contains the project report, results, and evaluation plots for the BERT + SPST fake news detection experiment. The full training code and reproduction pipeline live in the experiment repository.

---

## Results

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 97.52%  |
| Precision | 97.45%  |
| Recall    | 97.98%  |
| F1 Score  | 97.71%  |
| AUC-ROC   | 0.9981  |
| Test Loss | 0.0586  |

---

## What is SPST?

Standard BERT fine-tuning unfreezes all 110M parameters at once, requiring significant GPU memory. **Sequential Parameter Segment Training (SPST)** instead unfreezes the model layer-by-layer across four training segments — classifier head first, then progressively deeper encoder layers. This keeps peak VRAM low enough to train on a free Colab T4 GPU while still converging to a high-quality solution.

Without SPST: ~75% accuracy. With SPST: **97.52%**.

---

## Files

| File | Description |
|------|-------------|
| `SPST_Project_Report.docx` | Full project report — method, dataset, results, analysis |
| `results.json` | Raw metrics and per-epoch training history |
| `plots/` | Loss curves, F1 curves, confusion matrix, ROC curve, AUC curves |

---

## Model Weights

Trained weights, tokenizer, and checkpoints are available on Google Drive:

📁 **[Download Model Weights & Results](https://drive.google.com/drive/folders/1zAD6Q5RxQ-YGfiV455_DpG3_Wd2sgLkW?usp=sharing)**

To load the model:

```python
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("path/to/final_model")
tokenizer = BertTokenizer.from_pretrained("path/to/final_model")
```

---

## Reproduce the Experiment

The full pipeline — dataset scraping, preprocessing, and training — is in the experiment repository:

🔬 **[Experiment Repository](https://github.com/premananda-cloud/Bert_training_via_SPST)**

It includes the Colab notebook, all scraper scripts, the unify/clean pipeline, and step-by-step instructions to reproduce everything from scratch.

---

## Dataset

Three sources unified into a single corpus (80/10/10 train/val/test split):

| Dataset    | Fake | Real |
|------------|------|------|
| GossipCop  | 3000 | 3000 |
| PolitiFact | 500  | 500  |
| LIAR       | ~6300| ~6300|

📦 [Download Dataset ZIP](https://drive.google.com/file/d/1BfIvNcVJQIx8-KAqWFavy-htKcjTbMTb/view?usp=sharing)
