# BERT Training Using SPST: A Resource-Efficient Approach

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-blue)](./BERT_Fake_News_Detection_Research_Paper.pdf)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Code Repository](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/premananda-cloud/fake_news_detector)

## Fake News Detection Using BERT with Sequential Parameter Switch Training

This repository contains the research paper **"BERT Training Using SPST: A Resource-Efficient Approach"** by Mayanglambam Premananda Singh, demonstrating effective fake news detection within computational constraints.

---

## Paper Abstract

The proliferation of fake news poses a significant threat to information integrity in the digital age. This paper presents a novel approach to fake news detection using BERT (Bidirectional Encoder Representations from Transformers) with Sequential Parameter Switch Training (SPST), specifically designed for resource-constrained environments. Our method addresses the computational limitations of training large language models by implementing a three-phase sequential training strategy that progressively unfreezes model layers, enabling effective fine-tuning within Google Colab's free tier GPU constraints.

The model achieves **75.08% accuracy, 86.49% precision, 70.53% recall, 77.70% F1-score, and 83.17% AUC-ROC** on a unified dataset combining LIAR, ISOT, and FakeNewsNet, demonstrating competitive performance while maintaining computational efficiency.

---

## Related Repositories

| Repository | Purpose | Link |
| --- | --- | --- |
| **Paper Repository** (You are here) | Contains the research paper, citation info, and methodology description | Current repo |
| **Experimental Code Repository** | Complete implementation, training scripts, datasets, and reproduction instructions | [github.com/premananda-cloud/fake_news_detector](https://github.com/premananda-cloud/fake_news_detector) |

---

## Sequential Parameter Switch Training (SPST) Methodology

**SPST** is a three-phase training approach designed to enable BERT fine-tuning on resource-constrained hardware:

### Phase 1: Classifier-Only Training (2 epochs)
- Only the classification head is trainable
- All BERT encoder layers remain frozen
- Learning rate: 3e-5
- Minimal memory footprint (~0.001% of parameters trained)

### Phase 2: Top-Layer Fine-Tuning (2 epochs)
- Unfreezes layers 10-11 and the classifier
- Lower layers (0-9) remain frozen to preserve pre-trained representations
- Learning rate: 2e-5
- Moderate memory usage (~12.9% of parameters trained)

### Phase 3: Full Model Fine-Tuning (1 epoch)
- All BERT parameters become trainable
- Learning rate: 1e-5 (reduced to prevent catastrophic forgetting)
- Full memory footprint but limited to single epoch
- Enables global optimization across all layers

---

## Key Advantages of SPST

| Advantage | Impact |
| --- | --- |
| **Memory Efficiency** | Gradients computed only for active parameters, reducing peak memory by ~40% |
| **Training Stability** | Progressive unfreezing maintains pre-trained knowledge while enabling task-specific adaptation |
| **Accessibility** | Enables training on free-tier GPUs (Google Colab NVIDIA T4 with ~15GB RAM) |
| **Computational Efficiency** | Combines gradient checkpointing, mixed precision, and gradient accumulation |
| **Faster Convergence** | Task-specific features develop gradually, preventing instability from large learning rates |

---

## Performance Results

| Metric | Value |
| --- | --- |
| **Accuracy** | 75.08% |
| **Precision** | 86.49% |
| **Recall** | 70.53% |
| **F1-Score** | 77.70% |
| **AUC-ROC** | 83.17% |
| **Peak Memory** | < 12GB VRAM |

**Key Finding:** The 86.49% precision demonstrates exceptional reliability for real-world deployment, minimizing false accusations of misinformation against legitimate news sources.

---

## Memory Optimization Techniques

Beyond SPST, the approach integrates:

- **Gradient Checkpointing**: Reduces activation memory by ~40% through computation reuse
- **Mixed Precision Training**: Uses FP16 computation with FP32 stability, reducing memory by ~50%
- **Gradient Accumulation**: Simulates batch size 32 using micro-batches of 8
- **Efficient Data Loading**: Optimized PyTorch DataLoader configuration
- **Batch Optimization**: Batch size 8, max sequence length 128 tokens

---

## Model Architecture

- **Base Model**: BERT-base-uncased (110M parameters)
- **Layers**: 12 transformer layers
- **Hidden Dimensions**: 768
- **Attention Heads**: 12
- **Classification Head**: Dense layer (768 → 2) with softmax activation

---

## Dataset Information

The unified dataset combines three prominent fake news sources:

- **LIAR**: 12,836 short statements from PolitiFact.com
- **ISOT**: 44,898 balanced articles from Reuters and flagged unreliable sources
- **FakeNewsNet**: Articles from PolitiFact and GossipCop with social context

All datasets were standardized to binary labels (fake/true), deduplicated, and split into training (70%), validation (15%), and test (15%) sets.

---

## Reproducibility

All experimental code, data preprocessing scripts, training implementations, and evaluation protocols are maintained in a separate repository:

### **Complete Implementation Repository:**
[github.com/premananda-cloud/fake_news_detector](https://github.com/premananda-cloud/fake_news_detector)

This includes:
- Complete PyTorch implementation with SPST
- Data preprocessing and dataset unification scripts
- Google Colab notebooks for one-click reproduction
- Full hyperparameter configurations
- Model checkpoints and evaluation scripts
- Memory profiling and performance benchmarks

---

## Why SPST Matters

Traditional BERT fine-tuning requires substantial GPU memory, limiting access to researchers without expensive hardware. SPST democratizes access to advanced NLP capabilities by:

1. **Enabling training on free-tier cloud platforms** without expensive infrastructure
2. **Maintaining competitive performance** comparable to full fine-tuning approaches
3. **Providing a reproducible framework** for the research community
4. **Extending beyond fake news** to other NLP tasks requiring resource-efficient training

---

## Practical Applications

- **Hybrid Systems**: High precision (86.49%) enables automated flagging with human review for uncertain cases
- **Educational Use**: Complete reproducible codebase serves as teaching resource for NLP and deep learning courses
- **Rapid Prototyping**: Efficient training methodology allows quick iteration for customized systems
- **Real-World Deployment**: High precision minimizes false accusations against legitimate news sources

---

## Future Directions

- Multi-class classification for nuanced credibility assessment
- Hierarchical approaches for handling longer documents (>128 tokens)
- Multimodal integration with vision models for image/video analysis
- Cross-lingual support through multilingual BERT variants
- Explainability features for improved user trust and interpretability
- Continual learning to adapt to evolving misinformation tactics

---

## Citation

If you use this work in your research, please cite:

**APA Format:**
```
Singh, M. P. (2025). BERT Training Using SPST: A Resource-Efficient Approach. 
GitHub repository. https://github.com/premananda-cloud/Beert_fake_news_SPST
```

**BibTeX Format:**
```bibtex
@software{singh2025bert,
  author = {Singh, Mayanglambam Premananda},
  title = {BERT Training Using SPST: A Resource-Efficient Approach},
  year = {2025},
  url = {https://github.com/premananda-cloud/Beert_fake_news_SPST}
}
```

---

## License

This work is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

- Paper text: CC-BY-4.0 (sharing and adaptation with attribution)
- Code: MIT License

---

## Author

**Mayanglambam Premananda Singh**
- National Institute of Electronics and Information Technology, Imphal, Manipur
- Email: p.mangang@proton.me
- GitHub: [@premananda-cloud](https://github.com/premananda-cloud)

---

## Acknowledgments

- Google Colab for providing free GPU resources
- Hugging Face for the Transformers library
- Creators of LIAR, ISOT, and FakeNewsNet datasets
- Open-source ML community for supporting accessible research
