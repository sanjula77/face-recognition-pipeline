"""
Advanced Optuna-based hyperparameter tuning for face recognition classifiers
"""
import os
import sys
from pathlib import Path
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

class AdvancedOptunaTuner:
    """
    Advanced hyperparameter tuning using Optuna for face recognition classifiers.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, n_trials=20):
        """
        Initialize the tuner.
        
        Args:
            X_train: Training embeddings
            y_train: Training labels
            X_test: Test embeddings
            y_test: Test labels
            n_trials: Number of Optuna trials per algorithm
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        
        # Initialize MLflow if available
        if HAS_MLFLOW:
            try:
                from src.config import settings
                mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            except Exception:
                pass
    
    def svm_objective(self, trial):
        """Optuna objective for SVM hyperparameter tuning."""
        C = trial.suggest_float('C', 0.1, 100.0, log=True)
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto', 0.001, 0.01, 0.1, 1.0])
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
        
        model = SVC(C=C, gamma=gamma, kernel=kernel, probability=True, random_state=42)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, n_jobs=-1)
        
        return scores.mean()
    
    def knn_objective(self, trial):
        """Optuna objective for KNN hyperparameter tuning."""
        n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
        weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine'])
        
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, n_jobs=-1)
        
        return scores.mean()
    
    def random_forest_objective(self, trial):
        """Optuna objective for Random Forest hyperparameter tuning."""
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 5, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, n_jobs=-1)
        
        return scores.mean()
    
    def logistic_regression_objective(self, trial):
        """Optuna objective for Logistic Regression hyperparameter tuning."""
        C = trial.suggest_float('C', 0.01, 100.0, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
        max_iter = trial.suggest_int('max_iter', 100, 1000)
        
        model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, n_jobs=-1)
        
        return scores.mean()
    
    def _train_final_model(self, model_name, best_params):
        """
        Train final model with best parameters.
        
        Args:
            model_name: Name of the model ('SVM', 'KNN', 'RandomForest', 'LogisticRegression')
            best_params: Best hyperparameters from Optuna
        
        Returns:
            Trained model
        """
        if model_name == 'SVM':
            model = SVC(**best_params, probability=True, random_state=42)
        elif model_name == 'KNN':
            model = KNeighborsClassifier(**best_params)
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(**best_params, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        model.fit(self.X_train, self.y_train)
        return model
    
    def _log_to_mlflow(self, model_name, best_params, cv_score, test_accuracy):
        """
        Log results to MLflow if available.
        
        Args:
            model_name: Name of the model
            best_params: Best hyperparameters
            cv_score: Cross-validation score
            test_accuracy: Test accuracy
        """
        if not HAS_MLFLOW:
            return
        
        try:
            with mlflow.start_run(nested=True):
                mlflow.log_params({f"{model_name}_{k}": v for k, v in best_params.items()})
                mlflow.log_metric(f"{model_name}_cv_score", cv_score)
                mlflow.log_metric(f"{model_name}_test_accuracy", test_accuracy)
        except Exception:
            pass  # Continue if MLflow fails

