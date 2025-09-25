# Face Recognition Pipeline

## Overview
A modular face recognition system that supports:
- Data collection & preprocessing (alignment, CLAHE)
- Embeddings extraction (ArcFace, InsightFace)
- Classifier training & evaluation
- Inference API for recognition

## Project Structure
(Explain src/, data/, configs/, tests/)

## Setup
1. Clone repo
2. Create conda env
3. Install dependencies (`pip install -r requirements.txt`)
4. Create `.env` from `.env.example`

## Usage
- Preprocessing: `python src/preprocessing/pipeline.py`
- Training: `python src/training/train_classifier.py`
- Evaluation: `python src/evaluation/evaluate.py`
- Serve API: `uvicorn src.serving.api:app --reload`

## Contributing
Use feature branches, follow Conventional Commits.

## License
MIT
