#!/usr/bin/env python3
"""
CORRECTED Comprehensive comparison with proper train/test splits.
This fixes the data leakage issue that caused inflated 100% accuracy results.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import optuna
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train_classifier import (
    load_embeddings_from_dir, 
    precompute_augmented_embeddings,
    ArcFaceWrapper
)
from src.training.advanced_optuna import AdvancedOptunaTuner
from src.embeddings.normalization import EmbeddingNormalizer, get_best_normalization_method
from src.training.model_ensemble import ModelEnsemble
from src.training.confidence_calibration import ConfidenceCalibrator

class CorrectedComparison:
    """
    CORRECTED comparison system with proper train/test splits to avoid data leakage.
    """
    
    def __init__(self, results_dir="models/trained", n_trials=20,
                 use_ensemble=True, use_calibration=True, use_enhanced_normalization=True):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.n_trials = n_trials
        self.use_ensemble = use_ensemble
        self.use_calibration = use_calibration
        self.use_enhanced_normalization = use_enhanced_normalization
        self.normalizer = None
        
        # Create separate model directories
        self.embeddings_models_dir = self.results_dir / "embeddings_mode_models"
        self.images_models_dir = self.results_dir / "images_mode_models"
        self.embeddings_models_dir.mkdir(exist_ok=True)
        self.images_models_dir.mkdir(exist_ok=True)
        
        # Results storage
        self.results = {
            "embeddings_mode": {},
            "images_mode": {},
            "comparison": {},
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_trials": n_trials,
                "algorithms": ["SVM", "KNN", "RandomForest", "LogisticRegression"],
                "note": "CORRECTED VERSION - Proper train/test splits to avoid data leakage"
            }
        }
    
    def run_embeddings_mode_corrected(self, emb_dir="data/embeddings"):
        """
        Run corrected hyperparameter tuning on embeddings mode with proper train/test split.
        """
        print("🔍 EMBEDDINGS MODE - CORRECTED Hyperparameter Tuning")
        print("=" * 60)
        
        # Load embeddings data
        print(f"Loading embeddings from {emb_dir}...")
        X, y, names = load_embeddings_from_dir(emb_dir)
        print(f"Loaded {X.shape[0]} embeddings, dimension: {X.shape[1]}")
        
        # Convert string labels to numeric
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)
        print(f"Classes: {le.classes_}")
        print(f"Class distribution: {np.bincount(y_numeric)}")
        
        # Validate that we have at least 2 classes
        n_classes = len(le.classes_)
        if n_classes < 2:
            error_msg = (
                f"\n❌ ERROR: Cannot train classifier with only {n_classes} class(es)!\n"
                f"   Found classes: {list(le.classes_)}\n"
                f"   Total embeddings: {len(y)}\n\n"
                f"   Possible causes:\n"
                f"   1. Only one person's images were processed in preprocessing\n"
                f"   2. Face detection failed for other people's images\n"
                f"   3. Embedding extraction only found one person's files\n\n"
                f"   Please check:\n"
                f"   - data/processed/ directory - should have subdirectories for each person\n"
                f"   - data/embeddings/ directory - should have embeddings from multiple people\n"
                f"   - Run preprocessing again to ensure all people are processed\n"
            )
            raise ValueError(error_msg)
        
        # Normalization - enhanced or simple based on settings
        if hasattr(self, 'use_enhanced_normalization') and self.use_enhanced_normalization:
            # Enhanced normalization - find best method
            print("\n🔧 Finding best normalization method...")
            X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
                X, y_numeric, test_size=0.2, stratify=y_numeric, random_state=42
            )
            best_norm_method = get_best_normalization_method(X_train_temp, y_train_temp)
            
            # Apply best normalization
            normalizer = EmbeddingNormalizer(method=best_norm_method)
            normalizer.fit(X_train_temp)  # Fit on training data
            X_norm = normalizer.normalize(X)
            
            # Store normalizer for later use and save it
            self.normalizer = normalizer
            normalizer_path = self.embeddings_models_dir / 'normalizer.joblib'
            joblib.dump(normalizer, normalizer_path)
            print(f"✅ Normalizer saved: {normalizer_path}")
        else:
            # Simple L2 normalization (original)
            print("\n🔧 Using simple L2 normalization...")
            X_norm = normalize(X, axis=1)
            self.normalizer = None
        
        # PROPER train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y_numeric, test_size=0.2, stratify=y_numeric, random_state=42
        )
        
        print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
        print(f"Train classes: {np.bincount(y_train)}")
        print(f"Test classes: {np.bincount(y_test)}")
        
        # Run advanced Optuna tuning with CORRECTED data
        tuner = AdvancedOptunaTuner(
            X_train, y_train, X_test, y_test,  # PROPER train/test split
            n_trials=self.n_trials
        )
        
        # Run optimization with custom model paths
        results = self._run_optimization_with_custom_paths(
            tuner, 
            self.embeddings_models_dir,
            "embeddings",
            le.classes_
        )
        
        self.results["embeddings_mode"] = results
        
        # Create model ensemble if enabled
        if self.use_ensemble and len(results) > 1:
            print(f"\n🔧 Creating model ensemble...")
            ensemble = ModelEnsemble(voting_method='soft')
            ensemble.load_models_from_dir(self.embeddings_models_dir)
            
            # Evaluate ensemble
            ensemble_results = ensemble.evaluate_models(tuner.X_test, tuner.y_test)
            
            # Save ensemble
            ensemble_path = self.embeddings_models_dir / "ensemble.joblib"
            joblib.dump(ensemble, ensemble_path)
            print(f"✅ Ensemble saved: {ensemble_path}")
            
            self.results["embeddings_mode"]["ensemble"] = ensemble_results
        
        print(f"✅ Embeddings mode completed. Models saved to: {self.embeddings_models_dir}")
        return results
    
    def run_images_mode_corrected(self, images_dir="data/processed", n_augment=1, batch_size=16):
        """
        Run corrected hyperparameter tuning on images mode with proper train/test split.
        """
        print("\n🖼️  IMAGES MODE - CORRECTED Hyperparameter Tuning with Augmentation")
        print("=" * 60)
        
        # Build arcface wrapper and precompute embeddings
        print("Initializing ArcFace and precomputing augmented embeddings...")
        arcface = ArcFaceWrapper()
        
        print(f"Precomputing augmented embeddings (n_augment={n_augment})...")
        X, y = precompute_augmented_embeddings(
            arcface, images_dir, 
            n_augment=n_augment, 
            batch_size=batch_size
        )
        print(f"Precomputed {X.shape[0]} embeddings, dimension: {X.shape[1]}")
        
        # Convert string labels to numeric
        le = LabelEncoder()
        y_numeric = le.fit_transform(y)
        print(f"Classes: {le.classes_}")
        print(f"Class distribution: {np.bincount(y_numeric)}")
        
        # PROPER train/test split
        X_norm = normalize(X, axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y_numeric, test_size=0.2, stratify=y_numeric, random_state=42
        )
        
        print(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]} samples")
        print(f"Train classes: {np.bincount(y_train)}")
        print(f"Test classes: {np.bincount(y_test)}")
        
        # Run advanced Optuna tuning with CORRECTED data
        tuner = AdvancedOptunaTuner(
            X_train, y_train, X_test, y_test,  # PROPER train/test split
            n_trials=self.n_trials
        )
        
        # Run optimization with custom model paths
        results = self._run_optimization_with_custom_paths(
            tuner, 
            self.images_models_dir,
            "images",
            le.classes_
        )
        
        self.results["images_mode"] = results
        
        print(f"✅ Images mode completed. Models saved to: {self.images_models_dir}")
        return results
    
    def _run_optimization_with_custom_paths(self, tuner, models_dir, mode_name, class_names):
        """
        Run optimization with custom model saving paths and proper evaluation.
        """
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        
        sampler = TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            seed=42
        )
        
        pruner = MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1
        )
        
        models_to_optimize = {
            "SVM": tuner.svm_objective,
            "KNN": tuner.knn_objective,
            "RandomForest": tuner.random_forest_objective,
            "LogisticRegression": tuner.logistic_regression_objective
        }
        
        results = {}
        
        for model_name, objective_func in models_to_optimize.items():
            print(f"\n🔍 Optimizing {model_name} ({mode_name} mode)...")
            
            study = optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner
            )
            
            study.optimize(objective_func, n_trials=self.n_trials)
            
            # Check if any trials completed
            if len(study.trials) == 0 or all(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials):
                print(f"⚠️  All {model_name} trials were pruned. Skipping {model_name}.")
                continue
            
            # Train final model with best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            print(f"Best {model_name} params: {best_params}")
            print(f"Best {model_name} CV score: {best_score:.4f}")
            
            # Train and evaluate final model on PROPER test set
            final_model = tuner._train_final_model(model_name, best_params)
            
            # Apply confidence calibration if enabled
            y_pred = None
            if self.use_calibration:
                print(f"   🔧 Calibrating {model_name} probabilities...")
                y_proba_train = final_model.predict_proba(tuner.X_train)
                calibrator = ConfidenceCalibrator(method='isotonic')
                calibrator.fit(tuner.y_train, y_proba_train)
                
                # Test calibrated predictions
                y_proba_test = final_model.predict_proba(tuner.X_test)
                y_proba_calibrated = calibrator.calibrate_proba(y_proba_test)
                y_pred = np.argmax(y_proba_calibrated, axis=1)
                test_accuracy = np.mean(y_pred == tuner.y_test)
                
                # Save calibrator
                calibrator_path = models_dir / f"{model_name.lower()}_calibrator.joblib"
                joblib.dump(calibrator, calibrator_path)
                print(f"   ✅ Calibrator saved: {calibrator_path}")
            else:
                test_accuracy = final_model.score(tuner.X_test, tuner.y_test)
                y_pred = final_model.predict(tuner.X_test)
            
            # Get detailed classification report
            report = classification_report(tuner.y_test, y_pred, target_names=class_names, output_dict=True)
            
            print(f"{model_name} Test accuracy: {test_accuracy:.4f}")
            print(f"{model_name} Test precision: {report['macro avg']['precision']:.4f}")
            print(f"{model_name} Test recall: {report['macro avg']['recall']:.4f}")
            print(f"{model_name} Test F1-score: {report['macro avg']['f1-score']:.4f}")
            
            # Save model with custom path
            model_path = models_dir / f"{model_name.lower()}.joblib"
            joblib.dump(final_model, model_path)
            print(f"Saved {model_name} to {model_path}")
            
            results[model_name] = {
                "best_params": best_params,
                "cv_score": best_score,
                "test_accuracy": test_accuracy,
                "test_precision": report['macro avg']['precision'],
                "test_recall": report['macro avg']['recall'],
                "test_f1": report['macro avg']['f1-score'],
                "model_path": str(model_path),
                "n_trials": len(study.trials),
                "pruned_trials": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED),
                "classification_report": report
            }
            
            # Log to MLflow
            tuner._log_to_mlflow(model_name, best_params, best_score, test_accuracy)
        
        return results
    
    def compare_results(self):
        """
        Compare results between embeddings mode and images mode.
        """
        print("\n📊 CORRECTED COMPARISON ANALYSIS")
        print("=" * 60)
        
        embeddings_results = self.results["embeddings_mode"]
        images_results = self.results["images_mode"]
        
        # Create comparison table
        comparison_data = []
        
        algorithms = set(embeddings_results.keys()) | set(images_results.keys())
        
        for algorithm in algorithms:
            emb_data = embeddings_results.get(algorithm, {})
            img_data = images_results.get(algorithm, {})
            
            comparison_data.append({
                "Algorithm": algorithm,
                "Embeddings_CV_Score": emb_data.get("cv_score", 0),
                "Embeddings_Test_Accuracy": emb_data.get("test_accuracy", 0),
                "Embeddings_F1_Score": emb_data.get("test_f1", 0),
                "Images_CV_Score": img_data.get("cv_score", 0),
                "Images_Test_Accuracy": img_data.get("test_accuracy", 0),
                "Images_F1_Score": img_data.get("test_f1", 0),
                "Accuracy_Difference": img_data.get("test_accuracy", 0) - emb_data.get("test_accuracy", 0),
                "F1_Difference": img_data.get("test_f1", 0) - emb_data.get("test_f1", 0),
                "Best_Mode": "Images" if img_data.get("test_accuracy", 0) > emb_data.get("test_accuracy", 0) else "Embeddings"
            })
        
        # Create DataFrame for analysis
        df = pd.DataFrame(comparison_data)
        
        print("\n📈 CORRECTED COMPARISON TABLE:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Find best overall models
        best_embeddings = max(embeddings_results.items(), key=lambda x: x[1].get("test_accuracy", 0))
        best_images = max(images_results.items(), key=lambda x: x[1].get("test_accuracy", 0))
        
        print(f"\n🏆 BEST EMBEDDINGS MODE MODEL:")
        print(f"   Algorithm: {best_embeddings[0]}")
        print(f"   Test Accuracy: {best_embeddings[1]['test_accuracy']:.4f}")
        print(f"   Test F1-Score: {best_embeddings[1]['test_f1']:.4f}")
        print(f"   CV Score: {best_embeddings[1]['cv_score']:.4f}")
        
        print(f"\n🏆 BEST IMAGES MODE MODEL:")
        print(f"   Algorithm: {best_images[0]}")
        print(f"   Test Accuracy: {best_images[1]['test_accuracy']:.4f}")
        print(f"   Test F1-Score: {best_images[1]['test_f1']:.4f}")
        print(f"   CV Score: {best_images[1]['cv_score']:.4f}")
        
        # Overall winner
        if best_images[1]['test_accuracy'] > best_embeddings[1]['test_accuracy']:
            winner = "Images Mode"
            winner_model = best_images
        else:
            winner = "Embeddings Mode"
            winner_model = best_embeddings
        
        print(f"\n🎯 OVERALL WINNER: {winner}")
        print(f"   Best Algorithm: {winner_model[0]}")
        print(f"   Best Accuracy: {winner_model[1]['test_accuracy']:.4f}")
        print(f"   Best F1-Score: {winner_model[1]['test_f1']:.4f}")
        
        # Store comparison results
        self.results["comparison"] = {
            "comparison_table": comparison_data,
            "best_embeddings_model": {
                "algorithm": best_embeddings[0],
                "accuracy": best_embeddings[1]['test_accuracy'],
                "f1_score": best_embeddings[1]['test_f1'],
                "cv_score": best_embeddings[1]['cv_score']
            },
            "best_images_model": {
                "algorithm": best_images[0],
                "accuracy": best_images[1]['test_accuracy'],
                "f1_score": best_images[1]['test_f1'],
                "cv_score": best_images[1]['cv_score']
            },
            "overall_winner": {
                "mode": winner,
                "algorithm": winner_model[0],
                "accuracy": winner_model[1]['test_accuracy'],
                "f1_score": winner_model[1]['test_f1']
            }
        }
        
        return df
    
    def save_results(self):
        """
        Save all results to JSON and CSV files.
        """
        print(f"\n💾 SAVING CORRECTED RESULTS TO {self.results_dir}")
        print("=" * 60)
        
        # Save detailed results as JSON
        results_file = self.results_dir / "corrected_detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✅ Corrected detailed results saved to: {results_file}")
        
        # Save comparison table as CSV
        if "comparison" in self.results and "comparison_table" in self.results["comparison"]:
            comparison_file = self.results_dir / "corrected_comparison_table.csv"
            df = pd.DataFrame(self.results["comparison"]["comparison_table"])
            df.to_csv(comparison_file, index=False)
            print(f"✅ Corrected comparison table saved to: {comparison_file}")
        
        # Save summary report
        summary_file = self.results_dir / "corrected_summary_report.txt"
        with open(summary_file, 'w') as f:
            f.write("CORRECTED FACE RECOGNITION COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write("⚠️  IMPORTANT: This is the CORRECTED version with proper train/test splits!\n")
            f.write("The previous results were inflated due to data leakage.\n\n")
            f.write(f"Timestamp: {self.results['metadata']['timestamp']}\n")
            f.write(f"Number of trials per algorithm: {self.results['metadata']['n_trials']}\n")
            f.write(f"Algorithms tested: {', '.join(self.results['metadata']['algorithms'])}\n\n")
            
            if "comparison" in self.results:
                comp = self.results["comparison"]
                f.write("OVERALL WINNER:\n")
                f.write(f"  Mode: {comp['overall_winner']['mode']}\n")
                f.write(f"  Algorithm: {comp['overall_winner']['algorithm']}\n")
                f.write(f"  Accuracy: {comp['overall_winner']['accuracy']:.4f}\n")
                f.write(f"  F1-Score: {comp['overall_winner']['f1_score']:.4f}\n\n")
                
                f.write("BEST EMBEDDINGS MODE MODEL:\n")
                f.write(f"  Algorithm: {comp['best_embeddings_model']['algorithm']}\n")
                f.write(f"  Accuracy: {comp['best_embeddings_model']['accuracy']:.4f}\n")
                f.write(f"  F1-Score: {comp['best_embeddings_model']['f1_score']:.4f}\n")
                f.write(f"  CV Score: {comp['best_embeddings_model']['cv_score']:.4f}\n\n")
                
                f.write("BEST IMAGES MODE MODEL:\n")
                f.write(f"  Algorithm: {comp['best_images_model']['algorithm']}\n")
                f.write(f"  Accuracy: {comp['best_images_model']['accuracy']:.4f}\n")
                f.write(f"  F1-Score: {comp['best_images_model']['f1_score']:.4f}\n")
                f.write(f"  CV Score: {comp['best_images_model']['cv_score']:.4f}\n\n")
                
                f.write("MODEL LOCATIONS:\n")
                f.write(f"  Embeddings models: {self.embeddings_models_dir}\n")
                f.write(f"  Images models: {self.images_models_dir}\n")
        
        print(f"✅ Corrected summary report saved to: {summary_file}")
    
    def run_corrected_comparison(self, emb_dir="data/embeddings", images_dir="data/processed", 
                               n_augment=1, batch_size=16):
        """
        Run the CORRECTED comparison between embeddings and images mode.
        """
        start_time = time.time()
        
        print("🚀 STARTING CORRECTED FACE RECOGNITION COMPARISON")
        print("=" * 60)
        print("⚠️  IMPORTANT: This version fixes the data leakage issue!")
        print("   Previous 100% results were due to using same data for train/test.")
        print("=" * 60)
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Number of trials per algorithm: {self.n_trials}")
        print(f"Algorithms: SVM, KNN, RandomForest, LogisticRegression")
        print("=" * 60)
        
        # Run embeddings mode
        embeddings_results = self.run_embeddings_mode_corrected(emb_dir)
        
        # Run images mode
        images_results = self.run_images_mode_corrected(images_dir, n_augment, batch_size)
        
        # Compare results
        comparison_df = self.compare_results()
        
        # Save all results
        self.save_results()
        
        total_time = time.time() - start_time
        print(f"\n🎉 CORRECTED COMPARISON COMPLETED IN {total_time:.2f} SECONDS")
        print(f"📁 All corrected results saved to: {self.results_dir}")
        
        return {
            "embeddings_results": embeddings_results,
            "images_results": images_results,
            "comparison": comparison_df,
            "total_time": total_time
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="CORRECTED comparison of embeddings vs images mode")
    parser.add_argument("--emb_dir", default="data/embeddings", help="Embeddings directory")
    parser.add_argument("--images_dir", default="data/processed", help="Processed images directory")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials per algorithm")
    parser.add_argument("--n_augment", type=int, default=1, help="Number of augmentations for images mode")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for image processing")
    parser.add_argument("--results_dir", default="models/trained", help="Results directory")
    
    args = parser.parse_args()
    
    # Create corrected comparison system
    comparison = CorrectedComparison(
        results_dir=args.results_dir,
        n_trials=args.n_trials
    )
    
    # Run corrected comparison
    results = comparison.run_corrected_comparison(
        emb_dir=args.emb_dir,
        images_dir=args.images_dir,
        n_augment=args.n_augment,
        batch_size=args.batch_size
    )
    
    return results

if __name__ == "__main__":
    main()
