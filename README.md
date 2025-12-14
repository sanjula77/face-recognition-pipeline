# 🎯 Face Recognition System

A comprehensive, production-ready face recognition system supporting two powerful approaches: **Model-Based Recognition** (trained classifiers) and **One-Shot Learning** (template matching). Built with state-of-the-art deep learning models and optimized for accuracy and performance.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

---

## 📋 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [Documentation](#-documentation)
- [Contributing](#-contributing)

---

## ✨ Features

### 🎯 Dual Recognition Approaches

1. **Model-Based Recognition**
   - Trained machine learning classifiers (SVM, KNN, Random Forest, Logistic Regression)
   - Model ensemble support for improved accuracy
   - Confidence calibration for reliable predictions
   - Advanced embedding normalization
   - Face quality filtering

2. **One-Shot Learning**
   - Requires only **one reference image per person**
   - Uses RetinaFace for detection and ArcFace for embeddings
   - Cosine similarity matching
   - Fast setup, no training required
   - Ideal for small datasets or quick deployments

### 🔧 Advanced Capabilities

- **Face Detection**: RetinaFace (InsightFace) for robust face detection
- **Face Alignment**: Automatic face alignment and normalization
- **Preprocessing**: CLAHE enhancement, quality filtering, normalization
- **Embedding Extraction**: 512-dimensional ArcFace embeddings
- **Hyperparameter Tuning**: Optuna-based optimization
- **Batch Processing**: Process multiple images efficiently
- **Validation Tools**: Comprehensive test scripts with accuracy metrics

---

## 🏗️ System Architecture

### Model-Based Recognition Pipeline

```
Raw Images → Face Detection (RetinaFace) → Preprocessing (CLAHE, Quality Filter) 
→ Face Alignment → Embedding Extraction (ArcFace) → Model Training 
→ Hyperparameter Optimization → Model Ensemble → Production Model
```

### One-Shot Learning Pipeline

```
Reference Image → Face Detection (RetinaFace) → Face Alignment 
→ Embedding Extraction (ArcFace) → Database Storage

Query Image → Face Detection → Embedding Extraction 
→ Cosine Similarity Matching → Recognition Result
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd face-recognition-project
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download InsightFace Models

The InsightFace models will be automatically downloaded on first use. Alternatively, you can download them manually:

```bash
python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1)"
```

---

## 🚀 Quick Start

### Model-Based Recognition

#### 1. Prepare Your Dataset

Organize your images in the following structure:

```
data/raw/
├── person1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── person2/
│   ├── image1.jpg
│   └── ...
└── ...
```

#### 2. Run Training Pipeline

```bash
# Complete pipeline with validation
python scripts/pipeline/run_complete_pipeline.py

# Or standard pipeline (faster, no validation)
python scripts/pipeline/run_pipeline.py
```

#### 3. Recognize Faces

```bash
# Single image
python scripts/inference/face_recognizer.py data/test/person_1.jpg

# Or use test script
python tests/test_model_recognition.py
```

### One-Shot Learning

#### 1. Prepare Reference Images

Place one reference image per person:

```
data/reference_images/
├── person1.jpg
├── person2.jpg
└── ...
```

#### 2. Build Reference Database

```bash
python scripts/one_shot/build_reference_database.py --input_dir data/reference_images
```

#### 3. Recognize Faces

```bash
# Single image
python scripts/one_shot/recognize_one_shot.py --image data/test/person_1.jpg

# Group image (multiple faces)
python scripts/one_shot/recognize_one_shot.py --image data/test/group.jpg --group --show

# Or use test script
python tests/test_one_shot_recognition.py
```

---

## 📖 Usage

### Model-Based Recognition

#### Training

```bash
# Full pipeline with all steps
python scripts/pipeline/run_complete_pipeline.py

# Custom options
python scripts/pipeline/run_pipeline.py
```

#### Inference

```bash
# Using inference script
python scripts/inference/face_recognizer.py <image_path>

# Using test script (batch testing)
python tests/test_model_recognition.py --test_dir data/test/testUsingModel
```

### One-Shot Learning

#### Building Database

```bash
# Basic usage
python scripts/one_shot/build_reference_database.py --input_dir data/reference_images

# Custom database path
python scripts/one_shot/build_reference_database.py \
    --input_dir data/reference_images \
    --database_path my_database

# Use GPU (if available)
python scripts/one_shot/build_reference_database.py \
    --input_dir data/reference_images \
    --ctx_id 0
```

#### Recognition

```bash
# Single face
python scripts/one_shot/recognize_one_shot.py --image data/test/person.jpg

# Group image
python scripts/one_shot/recognize_one_shot.py \
    --image data/test/group.jpg \
    --group \
    --show

# Custom threshold
python scripts/one_shot/recognize_one_shot.py \
    --image data/test/person.jpg \
    --threshold 0.7
```

---

## 📁 Project Structure

```
face-recognition-project/
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
│
├── 📂 src/                         # Source code
│   ├── config.py                   # Configuration settings
│   ├── preprocessing/              # Face detection and preprocessing
│   │   ├── detect_align.py        # Face detection and alignment
│   │   ├── face_quality.py        # Quality assessment
│   │   └── pipeline.py            # Preprocessing pipeline
│   ├── embeddings/                 # Embedding extraction
│   │   ├── extractor.py           # Embedding extraction
│   │   ├── normalization.py      # Embedding normalization
│   │   └── utils.py               # Utility functions
│   ├── training/                   # Model training
│   │   ├── corrected_comparison.py # Main training script
│   │   ├── advanced_optuna.py     # Hyperparameter tuning
│   │   ├── confidence_calibration.py # Confidence calibration
│   │   ├── model_ensemble.py     # Model ensemble
│   │   └── train_classifier.py    # Classifier training
│   └── one_shot_recognition/       # One-shot learning
│       ├── database.py            # Reference database
│       ├── face_processor.py      # Face processing
│       ├── recognizer.py          # Recognition engine
│       └── similarity.py          # Similarity computation
│
├── 📂 scripts/                     # Executable scripts
│   ├── pipeline/                  # Training pipelines
│   │   ├── run_complete_pipeline.py # Full pipeline with validation
│   │   └── run_pipeline.py        # Standard pipeline
│   ├── one_shot/                  # One-shot learning scripts
│   │   ├── build_reference_database.py
│   │   └── recognize_one_shot.py
│   └── inference/                 # Inference scripts
│       └── face_recognizer.py
│
├── 📂 tests/                       # Test scripts
│   ├── test_model_recognition.py  # Model-based testing
│   ├── test_one_shot_recognition.py # One-shot testing
│   ├── test_single_image.py       # Single image test
│   └── test_group_image.py        # Group image test
│
├── 📂 data/                        # Data directory
│   ├── raw/                       # Raw training images
│   ├── processed/                  # Processed faces
│   ├── embeddings/                # Extracted embeddings
│   ├── reference_images/           # One-shot reference images
│   └── test/                      # Test images
│       ├── testUsingModel/        # Model test images
│       └── oneshortTest/          # One-shot test images
│
├── 📂 models/                      # Trained models
│   ├── production/                # Production-ready models
│   └── trained/                   # Training results
│
├── 📂 databases/                   # Database files
│   ├── reference_database/        # One-shot reference database
│   └── mlflow.db                  # MLflow tracking database
│
├── 📂 outputs/                     # Output files
│   ├── reports/                   # Analysis reports
│   └── visualizations/            # Charts and graphs
│
└── 📂 docs/                        # Documentation
    ├── PROJECT_STRUCTURE.md
    ├── INSIGHTFACE_EXPLANATION.md
    └── REORGANIZATION_SUMMARY.md
```

---

## 🧪 Testing

### Model-Based Recognition Test

```bash
# Test on default directory
python tests/test_model_recognition.py

# Custom test directory
python tests/test_model_recognition.py --test_dir data/test/testUsingModel

# Show detailed results
python tests/test_model_recognition.py --details

# Disable ensemble
python tests/test_model_recognition.py --no_ensemble
```

### One-Shot Recognition Test

```bash
# Test on default directory
python tests/test_one_shot_recognition.py

# Custom threshold
python tests/test_one_shot_recognition.py --threshold 0.7

# Show detailed results
python tests/test_one_shot_recognition.py --details
```

### Test Results

Both test scripts provide:
- ✅ Accuracy metrics
- ✅ Confidence/similarity scores
- ✅ Detailed prediction results
- ✅ Incorrect prediction analysis

---

## ⚙️ Configuration

### Model-Based Recognition

Configuration is managed in `src/config.py`:

```python
# Face detection settings
DETECTOR = "insightface"  # or "mtcnn"
OUTPUT_SIZE = (112, 112)
MIN_FACE_WIDTH_PX = 50

# CLAHE preprocessing
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
```

### One-Shot Learning

Default settings can be adjusted in scripts:

- **Similarity Threshold**: `--threshold` (default: 0.6)
- **Database Path**: `--database_path` (default: `databases/reference_database`)
- **Model**: `--model` (default: `buffalo_l`)

---

## 📚 Documentation

Additional documentation is available in the `docs/` directory:

- **[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Detailed project organization
- **[INSIGHTFACE_EXPLANATION.md](docs/INSIGHTFACE_EXPLANATION.md)** - InsightFace model details
- **[REORGANIZATION_SUMMARY.md](docs/REORGANIZATION_SUMMARY.md)** - Project reorganization details

---

## 🔄 Comparison: Model-Based vs One-Shot

| Feature | Model-Based | One-Shot Learning |
|---------|------------|-------------------|
| **Setup Time** | Longer (requires training) | Fast (no training) |
| **Data Requirements** | Multiple images per person | 1 image per person |
| **Accuracy** | High (trained on your data) | Good (uses pretrained models) |
| **Best For** | Large datasets, production | Quick setup, small datasets |
| **Flexibility** | Highly customizable | Simple and fast |
| **Maintenance** | Retrain when adding people | Just add reference image |

---

## 🛠️ Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
# Make sure you're in the project root directory
cd face-recognition-project
python scripts/pipeline/run_pipeline.py
```

**2. OpenMP Library Conflict**
- Already handled in the code with `KMP_DUPLICATE_LIB_OK=TRUE`

**3. CUDA/GPU Issues**
- The system defaults to CPU. For GPU support, ensure CUDA is properly installed.

**4. No Face Detected**
- Check image quality and lighting
- Ensure face is clearly visible
- Try different images

**5. Low Accuracy**
- For model-based: Ensure sufficient training data (10+ images per person)
- For one-shot: Use high-quality reference images with good lighting

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **InsightFace** - For RetinaFace detection and ArcFace embeddings
- **scikit-learn** - For machine learning models
- **OpenCV** - For image processing
- **Optuna** - For hyperparameter optimization

---

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**⭐ If you find this project useful, please consider giving it a star!**
