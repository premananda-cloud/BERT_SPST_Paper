# Fake News Detection Using BERT with Sequential Parameter Switch Training

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-blue)](BERT_Fake_News_Detection_Research_Paper.pdf)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)  <!-- Add after Zenodo archive -->
[![Code Repository](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/premananda-cloud/fake_news_detector)

# **A Resource-Efficient Approach for Training BERT in Constrained Environments**

This repository contains the research paper **"Fake News Detection Using BERT with Sequential Parameter Switch Training: A Resource-Efficient Approach"** by Mayanglambam Premananda Singh.

---

## 📄 Paper Abstract

The proliferation of fake news poses a significant threat to information integrity in the digital age. This paper presents a novel approach to fake news detection using BERT (Bidirectional Encoder Representations from Transformers) with Sequential Parameter Switch Training (SPST), specifically designed for resource-constrained environments. Our method addresses the computational limitations of training large language models by implementing a three-phase sequential training strategy that progressively unfreezes model layers, enabling effective fine-tuning within Google Colab's free tier GPU constraints.

The model achieves **75.08% accuracy, 86.49% precision, 70.53% recall, 77.70% F1-score, and 83.17% AUC-ROC** on a unified dataset combining LIAR, ISOT, and FakeNewsNet, demonstrating competitive performance while maintaining computational efficiency.

---

## 🔗 Related Repositories

| Repository | Purpose | Link |
|------------|---------|------|
| **Primary Paper Repository** (You are here) | Contains the research paper, citation info, and methodology description | Current repo |
| **Experimental Code Repository** | Complete implementation, training scripts, datasets, and reproduction instructions | [github.com/premananda-cloud/fake_news_detector](https://github.com/premananda-cloud/fake_news_detector) |

---

## 🏗️ Methodology Overview

**Sequential Parameter Switch Training (SPST)** is a three-phase training methodology:

1. **Phase 1 - Classifier-Only Training** (2 epochs): Only the classification head is trained
2. **Phase 2 - Top-Layer Fine-Tuning** (2 epochs): Layers 10-11 and classifier are unfrozen
3. **Phase 3 - Full Model Fine-Tuning** (1 epoch): All parameters are trainable

This approach reduces peak memory usage by **~40%** while maintaining competitive performance.

### Key Features:
- ✅ Progressive layer unfreezing for memory efficiency
- ✅ Gradient checkpointing and mixed precision training
- ✅ Optimized for Google Colab free tier (NVIDIA T4 GPU)
- ✅ Unified dataset from LIAR, ISOT, and FakeNewsNet

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 75.08% |
| **Precision** | 86.49% |
| **Recall** | 70.53% |
| **F1-Score** | 77.70% |
| **AUC-ROC** | 83.17% |
| **Training Memory** | < 12GB VRAM |

*Results achieved entirely within Google Colab free tier constraints.*

---

## 📂 Repository Contents
.
├── BERT_Fake_News_Detection_Research_Paper.pdf # Main research paper
├── README.md # This file
├── LICENSE # MIT License
└── CITATION.cff # Citation metadata (optional)

---

## 🧪 Reproducibility

All experimental code, data preprocessing scripts, training implementations, and evaluation protocols are maintained in a separate repository for clarity and maintainability:

### **Experimental Code Repository:**
**[github.com/premananda-cloud/fake_news_detector](https://github.com/premananda-cloud/fake_news_detector)**

This repository includes:
- Complete PyTorch implementation
- Data preprocessing and unification scripts
- Google Colab notebooks for one-click reproduction
- Configuration files with all hyperparameters
- Model checkpoints and evaluation scripts

---

## 📝 Citation

If you use this work in your research, please cite:

### **APA Format:**
Singh, M. P. (2025). Fake News Detection Using BERT with Sequential Parameter Switch Training: A Resource-Efficient Approach. *GitHub repository*. https://github.com/premananda-cloud/bert-spst-paper


# 📜 License
This work is licensed under the MIT License - see the LICENSE file for details.

The paper text is available under CC-BY-4.0 (allowing sharing and adaptation with attribution).

The code in the experimental repository is available under MIT License.

# 👤 Author
Mayanglambam Premananda Singh
National Institute of Electronics and Information Technology, Imphal, Manipur
Email: p.mangang@proton.me
GitHub: @premananda-cloud

# 🙏 Acknowledgments
Google Colab for providing free GPU resources

Hugging Face for the Transformers library

Creators of LIAR, ISOT, and FakeNewsNet datasets

Open-source ML community