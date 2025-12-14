"""
Confidence Calibration Module
Calibrates classifier probabilities to provide more reliable confidence scores
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression as CalibrationLR
from typing import Optional

class ConfidenceCalibrator:
    """
    Calibrates classifier probabilities for more reliable confidence scores.
    """
    
    def __init__(self, method='isotonic'):
        """
        Initialize calibrator.
        
        Args:
            method: Calibration method ('isotonic' or 'sigmoid')
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, y_true, y_proba):
        """
        Fit the calibrator on true labels and predicted probabilities.
        
        Args:
            y_true: True class labels
            y_proba: Predicted probabilities from classifier
        """
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:  # sigmoid
            self.calibrator = CalibrationLR()
        
        # Fit calibrator
        # For multi-class, we calibrate per class or use max probability
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            # Multi-class: calibrate the max probability
            max_proba = np.max(y_proba, axis=1)
            correct = (np.argmax(y_proba, axis=1) == y_true).astype(float)
        else:
            # Binary or single probability
            max_proba = y_proba.flatten() if y_proba.ndim > 1 else y_proba
            correct = (y_proba > 0.5).astype(float) if y_true.ndim == 1 else y_true
        
        self.calibrator.fit(max_proba.reshape(-1, 1), correct)
        self.is_fitted = True
    
    def calibrate_proba(self, y_proba):
        """
        Calibrate predicted probabilities.
        
        Args:
            y_proba: Raw probabilities from classifier
        
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            return y_proba
        
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            # Multi-class: calibrate and renormalize
            max_proba = np.max(y_proba, axis=1)
            calibrated_max = self.calibrator.predict(max_proba.reshape(-1, 1))
            
            # Renormalize probabilities
            calibrated_proba = y_proba.copy()
            max_indices = np.argmax(y_proba, axis=1)
            
            for i in range(len(calibrated_proba)):
                # Scale the max probability
                old_max = y_proba[i, max_indices[i]]
                if old_max > 0:
                    scale_factor = calibrated_max[i] / old_max
                    calibrated_proba[i] = y_proba[i] * scale_factor
                    # Renormalize to sum to 1
                    calibrated_proba[i] = calibrated_proba[i] / calibrated_proba[i].sum()
        else:
            # Binary or single probability
            proba_flat = y_proba.flatten() if y_proba.ndim > 1 else y_proba
            calibrated_flat = self.calibrator.predict(proba_flat.reshape(-1, 1))
            calibrated_proba = calibrated_flat.reshape(y_proba.shape) if y_proba.ndim > 1 else calibrated_flat
        
        return calibrated_proba

def calibrate_classifier(classifier, X_train, y_train, method='isotonic'):
    """
    Create a calibrated classifier wrapper.
    
    Args:
        classifier: Base classifier
        X_train: Training features
        y_train: Training labels
        method: Calibration method
    
    Returns:
        Calibrated classifier
    """
    calibrated = CalibratedClassifierCV(
        classifier, 
        method=method, 
        cv=5
    )
    calibrated.fit(X_train, y_train)
    return calibrated

