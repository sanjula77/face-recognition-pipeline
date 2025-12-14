"""
Model Ensemble Module
Combines multiple models for better accuracy and reliability
"""

import numpy as np
import joblib
from pathlib import Path
from typing import List, Dict, Optional, Literal
from sklearn.preprocessing import normalize

class ModelEnsemble:
    """
    Ensemble of multiple face recognition models.
    """
    
    def __init__(self, voting_method: Literal['soft', 'hard', 'weighted'] = 'soft'):
        """
        Initialize ensemble.
        
        Args:
            voting_method: How to combine predictions
                - 'soft': Average probabilities
                - 'hard': Majority vote
                - 'weighted': Weighted average based on model performance
        """
        self.voting_method = voting_method
        self.models = []
        self.model_names = []
        self.model_weights = []
        self.is_fitted = False
    
    def add_model(self, model, model_name: str, weight: float = 1.0):
        """
        Add a model to the ensemble.
        
        Args:
            model: Trained classifier model
            model_name: Name/identifier for the model
            weight: Weight for weighted voting (default: 1.0)
        """
        self.models.append(model)
        self.model_names.append(model_name)
        self.model_weights.append(weight)
        self.is_fitted = True
    
    def load_models_from_dir(self, models_dir: str, model_names: Optional[List[str]] = None):
        """
        Load multiple models from a directory.
        
        Args:
            models_dir: Directory containing model files
            model_names: List of model names to load (default: all .joblib files)
        """
        models_path = Path(models_dir)
        
        # Define actual model names (exclude calibrators, normalizers, and ensemble itself)
        actual_model_names = ['svm', 'knn', 'randomforest', 'logisticregression']
        
        if model_names is None:
            # Load only actual model files (not calibrators or normalizers)
            model_files = []
            for model_name in actual_model_names:
                model_file = models_path / f"{model_name}.joblib"
                if model_file.exists():
                    model_files.append(model_file)
        else:
            # Load specified models
            model_files = [models_path / f"{name.lower()}.joblib" for name in model_names]
        
        for model_file in model_files:
            if model_file.exists():
                # Skip if it's a calibrator, normalizer, or ensemble
                if '_calibrator' in model_file.stem or 'normalizer' in model_file.stem or 'ensemble' in model_file.stem:
                    continue
                
                try:
                    model = joblib.load(model_file)
                    # Check if it's actually a model (has predict_proba or predict method)
                    if not (hasattr(model, 'predict_proba') or hasattr(model, 'predict')):
                        print(f"⚠️  Skipping {model_file.name}: not a valid model")
                        continue
                    
                    model_name = model_file.stem
                    self.add_model(model, model_name)
                    print(f"✅ Loaded model: {model_name}")
                except Exception as e:
                    print(f"⚠️  Failed to load {model_file.name}: {e}")
            else:
                print(f"⚠️  Model not found: {model_file}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble predictions (probabilities).
        
        Args:
            X: Input features
        
        Returns:
            Ensemble probability predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble!")
        
        # Get predictions from all models
        all_probas = []
        for model in self.models:
            try:
                proba = model.predict_proba(X)
                all_probas.append(proba)
            except Exception as e:
                print(f"⚠️  Model prediction error: {e}")
                continue
        
        if not all_probas:
            raise ValueError("No valid model predictions!")
        
        # Combine predictions
        if self.voting_method == 'soft':
            # Average probabilities
            ensemble_proba = np.mean(all_probas, axis=0)
        
        elif self.voting_method == 'weighted':
            # Weighted average
            weights = np.array(self.model_weights[:len(all_probas)])
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_proba = np.zeros_like(all_probas[0])
            for i, proba in enumerate(all_probas):
                ensemble_proba += weights[i] * proba
        
        else:  # hard voting
            # Convert to class predictions, then back to probabilities
            predictions = [np.argmax(proba, axis=1) for proba in all_probas]
            predictions = np.array(predictions).T
            
            # Majority vote
            from scipy import stats
            majority = stats.mode(predictions, axis=1)[0].flatten()
            
            # Convert to probabilities (1.0 for predicted class, 0.0 for others)
            n_classes = all_probas[0].shape[1]
            ensemble_proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(majority):
                ensemble_proba[i, pred] = 1.0
        
        return ensemble_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble class predictions.
        
        Args:
            X: Input features
        
        Returns:
            Predicted class indices
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        """
        Get predictions with confidence scores.
        
        Args:
            X: Input features
        
        Returns:
            (predictions, confidences, probabilities)
        """
        proba = self.predict_proba(X)
        predictions = np.argmax(proba, axis=1)
        confidences = np.max(proba, axis=1)
        
        return predictions, confidences, proba
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate individual models and ensemble.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation results
        """
        results = {}
        
        # Evaluate individual models
        for i, (model, name) in enumerate(zip(self.models, self.model_names)):
            try:
                accuracy = model.score(X_test, y_test)
                results[name] = {
                    'accuracy': accuracy,
                    'weight': self.model_weights[i] if i < len(self.model_weights) else 1.0
                }
                print(f"   {name:20} accuracy: {accuracy:.4f}")
            except Exception as e:
                print(f"   {name:20} error: {e}")
                results[name] = {'accuracy': 0.0, 'error': str(e)}
        
        # Evaluate ensemble
        ensemble_pred = self.predict(X_test)
        ensemble_accuracy = np.mean(ensemble_pred == y_test)
        results['ensemble'] = {'accuracy': ensemble_accuracy}
        print(f"   {'ensemble':20} accuracy: {ensemble_accuracy:.4f}")
        
        return results
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """
        Optimize model weights based on validation performance.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
        """
        from scipy.optimize import minimize
        
        # Get predictions from all models
        all_probas = []
        for model in self.models:
            try:
                proba = model.predict_proba(X_val)
                all_probas.append(proba)
            except:
                continue
        
        if not all_probas:
            return
        
        # Objective function: minimize negative accuracy
        def objective(weights):
            weights = weights / weights.sum()  # Normalize
            ensemble_proba = np.zeros_like(all_probas[0])
            for i, proba in enumerate(all_probas):
                ensemble_proba += weights[i] * proba
            
            predictions = np.argmax(ensemble_proba, axis=1)
            accuracy = np.mean(predictions == y_val)
            return -accuracy  # Minimize negative accuracy
        
        # Optimize weights
        initial_weights = np.ones(len(all_probas)) / len(all_probas)
        bounds = [(0, 1) for _ in range(len(all_probas))]
        
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds)
        
        if result.success:
            self.model_weights = result.x.tolist()
            print(f"✅ Optimized weights: {dict(zip(self.model_names[:len(self.model_weights)], self.model_weights))}")
        else:
            print("⚠️  Weight optimization failed, using equal weights")

