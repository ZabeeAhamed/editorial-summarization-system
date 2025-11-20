# 📰 Editorial Summarization System

**AI-Powered Automatic Summarization for Indian Newspaper Editorials**

> **273% improvement in ROUGE-2 score** through domain adaptation of T5-small on Indian editorial content

**Author:** [ZabeeAhamed](https://github.com/ZabeeAhamed)  
**Date:** November 2025  
**Last Updated:** 2025-11-20

---

## 🎯 Overview

An end-to-end NLP system that automatically summarizes Indian newspaper editorials using a fine-tuned T5-small transformer model. The system demonstrates significant performance improvement through domain adaptation, achieving **273% better ROUGE-2 scores** compared to the baseline pretrained model.

### ✨ Key Highlights

- 🚀 **273% ROUGE-2 improvement** (0.1364 → 0.5094)
- 📊 **87 high-quality** article-summary pairs from real newspapers
- ⚡ **Fast training:** 25 seconds on GPU for 5 epochs
- 🌐 **Interactive web demo** with Streamlit
- 📈 **Production-ready** with comprehensive evaluation

---

## 📊 Performance Metrics

| Metric | Pretrained T5-small | Fine-tuned Model | **Improvement** |
|--------|---------------------|------------------|-----------------|
| **ROUGE-1** | 0.2229 | **0.5713** | **+156.33%** 📈 |
| **ROUGE-2** | 0.1364 | **0.5094** | **+273.45%** 🔥 |
| **ROUGE-L** | 0.1886 | **0.5407** | **+186.72%** ⭐ |

**Training Details:**
- Dataset: 87 samples (69 train, 8 val, 10 test)
- Epochs: 5
- Training time: ~25 seconds on CUDA GPU
- Best validation loss: 0.8122
- Smooth learning curve with no overfitting

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works)
- 4GB+ RAM

### Installation

```bash
# Clone repository
git clone https://github.com/ZabeeAhamed/editorial-summarization-system.git
cd editorial-summarization-system

# Create virtual environment 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Model Setup

⚠️ **Important:** Model files (~230MB) are **not included** in this repository due to GitHub's file size limits.

 **Train the Model Yourself**

```bash
# Requires dataset in data/ folder
python scripts/train_model.py

# Training takes:
# - ~25 seconds on GPU
# - ~1.5 hours on CPU

# Model will be saved to: models/finetuned_editorial_t5/
```


### Run the Demo

```bash
cd app
streamlit run demo.py
```

The web interface will open at: `http://localhost:8501`

---

## 📁 Project Structure

```
editorial-summarization-system/
├── app/
│   └── demo.py                      # Streamlit web interface
│
├── scripts/
│   ├── create_dataset.py            # OCR extraction & dataset creation
│   ├── train_model.py               # Fine-tuning pipeline ⭐
│   └── evaluate_model.py            # ROUGE evaluation
│
├── models/
│   └── .gitkeep                     # (Models excluded - see Model Setup)
│
├── dataset_output/
│   └── .gitkeep                     # (Dataset excluded - too large
│
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules

Note: Large files (models, datasets, images) excluded from repository.
```

---

## 🎓 Methodology

### 1. Data Collection

- **Source:** 196 Indian newspaper editorial images (scanned/photographed)
- **OCR Tool:** Tesseract with OpenCV preprocessing
- **Processing Pipeline:**
  - Image enhancement (denoising, contrast adjustment, sharpening)
  - Text extraction via OCR
  - Quality validation (word count, compression ratio)
  - Manual review and cleaning

### 2. Dataset Statistics

| Attribute | Value |
|-----------|-------|
| **Total samples** | 87 article-summary pairs |
| **Train/Val/Test split** | 80% / 10% / 10% |
| **Average article length** | 462 words |
| **Average summary length** | 128 words |
| **Compression ratio** | 0.29 (29% of original) |
| **Success rate** | 44.4% (87/196 images) |

### 3. Model Fine-tuning

- **Base Model:** T5-small (60M parameters, pretrained on C4)
- **Approach:** Domain adaptation via supervised fine-tuning
- **Loss Function:** Cross-entropy
- **Optimizer:** AdamW with linear warmup
- **Hyperparameters:**
  - Learning rate: 3e-4
  - Batch size: 4
  - Max input length: 512 tokens
  - Max output length: 150 tokens
  - Beam search: 4 beams

### 4. Evaluation

- **Metrics:** ROUGE-1, ROUGE-2, ROUGE-L (F1 scores)
- **Baseline:** Pretrained T5-small without fine-tuning
- **Test Set:** 10 held-out samples (never seen during training)

---

### Web Interface

1. Launch the Streamlit app:
   ```bash
   cd app
   streamlit run demo.py
   ```

2. Open browser at `http://localhost:8501`

3. Enter article text or click "Load Sample Article"

4. Click "Generate Summary" to see results

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Deep Learning** | PyTorch 2.0+ |
| **NLP Framework** | Hugging Face Transformers |
| **Model** | T5-small (60M parameters) |
| **OCR** | Tesseract 4.0+ |
| **Evaluation** | rouge-score |
| **Web Framework** | Streamlit |
| **Data Processing** | pandas, NumPy |

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **Test Set Distribution:** Test samples from same distribution as training data (same OCR quality, editorial style)
2. **Dataset Size:** 87 samples (typical research uses 100-500+)
3. **OCR Artifacts:** Source text contains OCR errors from image extraction
4. **Domain Specificity:** Optimized for Indian newspaper editorials
5. **Language:** English-only (no multilingual support)
6. **No Cross-Validation:** Single train/val/test split

### Planned Improvements

- [ ] Evaluate on completely fresh newspaper sources
- [ ] Implement k-fold cross-validation
- [ ] Expand dataset to 500+ samples
- [ ] Add multilingual support (Hindi, Tamil, etc.)
- [ ] Deploy as REST API (FastAPI)
- [ ] Test on different editorial styles (US, UK news)
- [ ] Implement abstractive summarization refinement
- [ ] Add BERT-based extractive summarization for comparison

---

## 📈 Training Results

### Loss Curves

| Epoch | Training Loss | Validation Loss | Status |
|-------|---------------|-----------------|--------|
| 1 | 3.5425 | 2.2220 | Initial learning |
| 2 | 1.9633 | 1.3725 | Strong improvement |
| 3 | 1.3367 | 0.9666 | Convergence |
| 4 | 1.0026 | 0.8577 | Fine-tuning |
| 5 | 0.9107 | **0.8122** | **Best model** ✅ |

**Key Observations:**
- ✅ Consistent decrease in both training and validation loss
- ✅ No evidence of overfitting
- ✅ Validation loss improvement of 63% (2.22 → 0.81)

---

##  Reproducibility

### To Reproduce Results

1. **Dataset Preparation:**
   - Collect 200+ newspaper editorial images
   - Run OCR extraction: `python scripts/create_dataset.py`
   - Expected: ~40-50% success rate

2. **Model Training:**
   ```bash
   python scripts/train_model.py
   ```
   - Uses same hyperparameters as reported
   - Random seed: 42
   - Expected training time: 25s (GPU)

3. **Evaluation:**
   ```bash
   python scripts/evaluate_model.py
   ```
   - Generates ROUGE scores
   - Compares with baseline

**Note:** Results may vary slightly due to random initialization and hardware differences.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation

---

## 👨‍💻 Author

**ZabeeAhamed**

- GitHub: [@ZabeeAhamed](https://github.com/ZabeeAhamed)
- LinkedIn: [ZabeeAhamed](https://linkedin.com/in/zabeeahmed)
- Email: zabeeahamed628@gmail.com

---

## 🙏 Acknowledgments

- Google Research for the T5 model architecture
- Hugging Face for the Transformers library
- Tesseract OCR development team
- Indian newspaper publications for editorial content

---


<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by ZabeeAhmed

🚀 Fine-tuned T5 | 📊 273% ROUGE-2 Improvement | ⚡ 25s Training

</div>
