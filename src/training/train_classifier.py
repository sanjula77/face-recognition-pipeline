# src/training/train_classifier.py
import os
import argparse
import glob
import numpy as np
import joblib
import mlflow
import sklearn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import optuna
import warnings
warnings.filterwarnings("ignore")

# local imports
try:
    from src.training.augs import get_train_transforms, get_val_transforms
    from src.training.dataset_images import FaceImageDataset, label_from_filename
    from src.training.advanced_optuna import AdvancedOptunaTuner
except ImportError:
    # Fallback for when running from project root
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from augs import get_train_transforms, get_val_transforms
    from dataset_images import FaceImageDataset, label_from_filename
    from advanced_optuna import AdvancedOptunaTuner
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Attempt to import arcface extractor (insightface) fallback to facenet-pytorch
HAS_INSIGHTFACE = False
HAS_FACENET = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except Exception:
    HAS_INSIGHTFACE = False

try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    HAS_FACENET = True
except Exception:
    HAS_FACENET = False

# Default directories
EMBEDDINGS_DIR = "data/embeddings"      # for mode=embeddings: .npy files
PROCESSED_IMG_DIR = "data/processed"    # for mode=images: aligned PNGs
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def l2_normalize(x, axis=1, eps=1e-10):
    return x / np.linalg.norm(x, axis=axis, keepdims=True).clip(min=eps)

# ---------------------------
# Embedding utilities
# ---------------------------
class ArcFaceWrapper:
    """Thin wrapper to provide batch embedding extraction. Tries insightface, otherwise uses facenet-pytorch."""
    def __init__(self, ctx_id=0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.ctx_id = ctx_id
        self.model = None
        if HAS_INSIGHTFACE:
            print("Using insightface for embedding extraction.")
            self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
            # ctx_id: 0 for GPU if available, -1 for CPU
            try:
                self.app.prepare(ctx_id=0, det_size=(112,112))
            except Exception:
                self.app.prepare(ctx_id=-1, det_size=(112,112))
            # Note: app.get expects BGR images and will run detection; but we pass already aligned crops
        elif HAS_FACENET:
            print("Using facenet-pytorch (InceptionResnetV1) fallback.")
            self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        else:
            raise RuntimeError("No embedding model available: install insightface or facenet-pytorch")

    def get_embeddings_from_images(self, batch_images):
        """
        Input: list or numpy array of images in BGR or RGB? 
        We assume batch_images is numpy array (B, H, W, C) in BGR (cv2 read).
        Returns: numpy array (B, 512) embeddings (float32)
        """
        if HAS_INSIGHTFACE:
            # Use direct recognition model for pre-aligned faces (same as embeddings mode)
            outs = []
            for img in batch_images:
                try:
                    # Use recognition model directly for pre-aligned faces
                    embedding = self.app.models['recognition'].get_feat(img)
                    # Flatten embedding to 1D vector
                    emb = embedding.flatten().astype(np.float32)
                except Exception as e:
                    # fallback zero vector
                    emb = np.zeros((512,), dtype=np.float32)
                outs.append(emb)
            emb_arr = np.stack(outs, axis=0)
            return emb_arr
        else:
            # facenet-pytorch expects RGB float tensors normalized to [-1,1]
            imgs = []
            for img in batch_images:
                # img is BGR; convert to RGB and scale to [-1,1]
                rgb = img[..., ::-1]  # BGR->RGB
                arr = rgb.astype('float32') / 255.0
                arr = (arr - 0.5) / 0.5  # [-1,1]
                # to CHW
                arr = np.transpose(arr, (2,0,1))
                imgs.append(arr)
            batch = torch.tensor(np.stack(imgs, axis=0), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                out = self.model(batch)
            out = out.cpu().numpy()
            return out

# ---------------------------
# Loading embeddings from disk
# ---------------------------
def load_embeddings_from_dir(emb_dir):
    """
    Expect files: data/embeddings/<label>_whatever.npy OR <filename>.npy where filename contains label prefix.
    Returns: X (N,512), y (N,), filenames (N,)
    """
    files = sorted(Path(emb_dir).glob("*.npy"))
    X=[]
    y=[]
    names=[]
    for p in files:
        arr = np.load(str(p))
        if arr.ndim==2:  # maybe (1,512)
            arr = arr.reshape(-1)
        if arr.size not in (128,256,512):
            # tolerate non standard dims
            pass
        label = Path(p).stem.split("_")[0]
        X.append(arr.astype(np.float32))
        y.append(label)
        names.append(str(p.name))
    if len(X)==0:
        raise RuntimeError(f"No embeddings found in {emb_dir}")
    X = np.stack(X, axis=0)
    return X, np.array(y), names

# ---------------------------
# Optuna objective for SVM (returns mean CV score)
# ---------------------------
def svm_objective(trial, X, y):
    C = trial.suggest_loguniform("C", 1e-3, 1e3)
    clf = SVC(C=C, kernel='linear', probability=True, random_state=42)
    
    # Adjust CV strategy based on dataset size
    unique_classes, counts = np.unique(y, return_counts=True)
    min_samples_per_class = counts.min()
    
    if min_samples_per_class < 2:
        # Use simple train/test split for very small datasets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_val, y_val)
        return float(score)
    else:
        # Use cross-validation for larger datasets
        n_splits = min(3, min_samples_per_class)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        return float(scores.mean())

def knn_objective(trial, X, y):
    # Adjust k based on dataset size
    n_samples = len(X)
    max_k = min(15, n_samples - 1)  # k must be less than n_samples
    k = trial.suggest_int("k", 1, max_k)
    
    metric = trial.suggest_categorical("metric", ["cosine","euclidean"])
    if metric=="cosine":
        # sklearn's KNeighbors doesn't have 'cosine' directly for algorithm='auto' prior to some versions.
        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    else:
        clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    
    # Adjust CV strategy based on dataset size
    unique_classes, counts = np.unique(y, return_counts=True)
    min_samples_per_class = counts.min()
    
    if min_samples_per_class < 2:
        # Use simple train/test split for very small datasets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
        clf.fit(X_train, y_train)
        score = clf.score(X_val, y_val)
        return float(score)
    else:
        # Use cross-validation for larger datasets
        n_splits = min(3, min_samples_per_class)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=1)
        return float(scores.mean())

# ---------------------------
# Main training routine
# ---------------------------
def train_on_embeddings(X, y, out_prefix="models/face_clf", n_trials=20):
    # L2 normalize
    Xn = l2_normalize(X, axis=1)
    
    # Check if we have enough samples per class for stratified split
    unique_classes, counts = np.unique(y, return_counts=True)
    min_samples_per_class = counts.min()
    
    if min_samples_per_class < 2:
        print(f"Warning: Some classes have only {min_samples_per_class} sample(s). Using simple split without stratification.")
        # Use simple split without stratification
        X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.20, random_state=42)
    else:
        # Use stratified split
        X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.20, stratify=y, random_state=42)

    # Optuna for SVM
    study_svm = optuna.create_study(direction="maximize")
    study_svm.optimize(lambda t: svm_objective(t, X_train, y_train), n_trials=n_trials)
    best_svm = study_svm.best_params
    print("Best SVM params:", best_svm)

    # Train final SVM
    svm = SVC(C=best_svm["C"], kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    preds = svm.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print("SVM Test accuracy:", acc)
    print(classification_report(y_test, preds))

    joblib.dump(svm, out_prefix + "_svm.joblib")
    print("Saved SVM to", out_prefix + "_svm.joblib")

    # KNN Optuna
    study_knn = optuna.create_study(direction="maximize")
    study_knn.optimize(lambda t: knn_objective(t, X_train, y_train), n_trials=n_trials)
    best_knn = study_knn.best_params
    print("Best KNN params:", best_knn)
    if best_knn["metric"] == "cosine":
        knn = KNeighborsClassifier(n_neighbors=best_knn["k"], metric="cosine")
    else:
        knn = KNeighborsClassifier(n_neighbors=best_knn["k"], metric="euclidean")
    knn.fit(X_train, y_train)
    preds_knn = knn.predict(X_test)
    acc_knn = accuracy_score(y_test, preds_knn)
    print("KNN Test accuracy:", acc_knn)
    print(classification_report(y_test, preds_knn))
    joblib.dump(knn, out_prefix + "_knn.joblib")
    print("Saved KNN to", out_prefix + "_knn.joblib")

    # Log to MLflow
    mlflow.set_tracking_uri("file:///mlflow.db")
    with mlflow.start_run(run_name="classifier_training"):
        mlflow.log_param("n_train", X_train.shape[0])
        mlflow.log_param("n_test", X_test.shape[0])
        mlflow.log_param("svm_C", float(best_svm["C"]))
        mlflow.log_metric("svm_acc", float(acc))
        mlflow.log_param("knn_k", int(best_knn["k"]))
        mlflow.log_param("knn_metric", best_knn["metric"])
        mlflow.log_metric("knn_acc", float(acc_knn))
        mlflow.log_artifact(out_prefix + "_svm.joblib")
        mlflow.log_artifact(out_prefix + "_knn.joblib")

    return {"svm_acc": float(acc), "knn_acc": float(acc_knn)}

def train_with_advanced_optuna(X, y, out_prefix="models/face_clf", n_trials=50, use_pruning=True):
    """
    Advanced training with Optuna using multiple algorithms and advanced features.
    """
    # L2 normalize
    Xn = l2_normalize(X, axis=1)
    
    # Check if we have enough samples per class for stratified split
    unique_classes, counts = np.unique(y, return_counts=True)
    min_samples_per_class = counts.min()
    
    if min_samples_per_class < 2:
        print(f"Warning: Some classes have only {min_samples_per_class} sample(s). Using simple split without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.20, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.20, stratify=y, random_state=42)
    
    print(f"🚀 Starting Advanced Optuna Hyperparameter Tuning")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Number of trials: {n_trials}")
    print(f"   Pruning enabled: {use_pruning}")
    print("=" * 60)
    
    # Initialize advanced tuner
    tuner = AdvancedOptunaTuner(X_train, y_train, X_test, y_test, n_trials=n_trials)
    
    # Run optimization
    results = tuner.optimize_all_models(use_pruning=use_pruning)
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]["test_accuracy"])
    best_name, best_result = best_model
    
    print(f"\n🏆 BEST MODEL: {best_name}")
    print(f"   Test Accuracy: {best_result['test_accuracy']:.4f}")
    print(f"   CV Score: {best_result['cv_score']:.4f}")
    print(f"   Best Parameters: {best_result['best_params']}")
    
    return results

def precompute_augmented_embeddings(arcface, images_dir, n_augment=5, batch_size=16):
    """
    Create (X,y) arrays by applying augmentations to images on the fly and extracting embeddings via arcface wrapper.
    Returns X (N,512), y (N,)
    """
    ds = FaceImageDataset(images_dir, transform=get_train_transforms())
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    all_emb = []
    all_labels = []
    temp_batch_imgs = []
    temp_batch_labels = []
    for (img, label, path) in loader:
        # img is numpy image (H,W,C) returned as a single item list because batch_size=1
        img = img[0].numpy() if hasattr(img[0], 'numpy') else img[0]  # but our dataset returns numpy
        # produce n_augment variants for this image
        variants = []
        variants.append(img)  # 1 original (transforms may have already been applied)
        for _ in range(n_augment-1):
            # re-apply transform by calling ds.__getitem__ on current index
            # easier approach: call the dataset transform manually
            if ds.transform:
                v = ds.transform(image=img)['image']
            else:
                v = img
            variants.append(v)
        # process variants in batches through arcface
        for i in range(0, len(variants), batch_size):
            batch_imgs = variants[i:i+batch_size]
            emb = arcface.get_embeddings_from_images(batch_imgs)
            for e in emb:
                all_emb.append(e.astype(np.float32))
                # Extract person name from filename instead of using dataset's idx2label
                filename = Path(path[0]).name
                person_name = label_from_filename(filename)
                all_labels.append(person_name)
    X = np.stack(all_emb, axis=0)
    y = np.array(all_labels, dtype=object)
    return X, y

def main(args):
    mlflow.set_tracking_uri("file:///mlflow.db")
    mode = args.mode
    if mode == "embeddings":
        print("Loading embeddings from", args.emb_dir)
        X, y, names = load_embeddings_from_dir(args.emb_dir)
        print("Loaded", X.shape[0], "embeddings, dim=", X.shape[1])
        
        if args.advanced:
            use_pruning = not args.no_pruning
            metrics = train_with_advanced_optuna(X, y, out_prefix=args.out_prefix, n_trials=args.n_trials, use_pruning=use_pruning)
        else:
            metrics = train_on_embeddings(X, y, out_prefix=args.out_prefix, n_trials=args.n_trials)
        print("Training finished. Metrics:", metrics)
        
    elif mode == "images":
        # build arcface wrapper
        arcface = ArcFaceWrapper()
        # precompute a dataset of augmented embeddings
        print("Precomputing augmented embeddings (this may take a while)...")
        X, y = precompute_augmented_embeddings(arcface, args.images_dir, n_augment=args.n_augment, batch_size=args.batch_size)
        print("Precomputed embeddings:", X.shape)
        
        if args.advanced:
            use_pruning = not args.no_pruning
            metrics = train_with_advanced_optuna(X, y, out_prefix=args.out_prefix, n_trials=args.n_trials, use_pruning=use_pruning)
        else:
            metrics = train_on_embeddings(X, y, out_prefix=args.out_prefix, n_trials=args.n_trials)
        print("Training finished. Metrics:", metrics)
    else:
        raise ValueError("mode must be 'embeddings' or 'images'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["embeddings","images"], default="embeddings")
    parser.add_argument("--emb_dir", default=EMBEDDINGS_DIR)
    parser.add_argument("--images_dir", default=PROCESSED_IMG_DIR)
    parser.add_argument("--out_prefix", default=os.path.join(MODELS_DIR, "face_clf"))
    parser.add_argument("--n_trials", type=int, default=20, help="Optuna trials for hyperparam search")
    parser.add_argument("--n_augment", type=int, default=5, help="Number of augmented variants per image for mode=images")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--advanced", action="store_true", help="Use advanced Optuna tuning with multiple algorithms")
    parser.add_argument("--no_pruning", action="store_true", help="Disable pruning in advanced mode")
    args = parser.parse_args()
    main(args)
