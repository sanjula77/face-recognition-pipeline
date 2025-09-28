#!/usr/bin/env python3
"""
Comprehensive comparison of embeddings mode vs images mode with advanced hyperparameter tuning.
Tests multiple algorithms and saves models in separate directories for easy comparison.
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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.train_classifier import (
    load_embeddings_from_dir, 
    precompute_augmented_embeddings,
    ArcFaceWrapper
)
from src.training.advanced_optuna import AdvancedOptunaTuner

class ComprehensiveComparison:
    """
    Comprehensive comparison system for embeddings vs images mode with advanced hyperparameter tuning.
    """
    
    def __init__(self, results_dir="comparison_results", n_trials=20):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.n_trials = n_trials
        
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
                "algorithms": ["SVM", "KNN", "RandomForest", "LogisticRegression"]
            }
        }
    
    def run_embeddings_mode_comparison(self, emb_dir="data/embeddings"):
        """
        Run advanced hyperparameter tuning on embeddings mode.
        """
        print("🔍 EMBEDDINGS MODE - Advanced Hyperparameter Tuning")
        print("=" * 60)
        
        # Load embeddings data
        print(f"Loading embeddings from {emb_dir}...")
        X, y, names = load_embeddings_from_dir(emb_dir)
        print(f"Loaded {X.shape[0]} embeddings, dimension: {X.shape[1]}")
        print(f"Classes: {sorted(set(names))}")
        
        # Run advanced Optuna tuning
        tuner = AdvancedOptunaTuner(
            X, y, X, y,  # Using same data for train/test in embeddings mode
            n_trials=self.n_trials
        )
        
        # Override model saving directory
        original_models_dir = "models"
        tuner.results = {}
        
        # Run optimization with custom model paths
        results = self._run_optimization_with_custom_paths(
            tuner, 
            self.embeddings_models_dir,
            "embeddings"
        )
        
        self.results["embeddings_mode"] = results
        
        print(f"✅ Embeddings mode completed. Models saved to: {self.embeddings_models_dir}")
        return results
    
    def run_images_mode_comparison(self, images_dir="data/processed", n_augment=1, batch_size=16):
        """
        Run advanced hyperparameter tuning on images mode with augmentation.
        """
        print("\n🖼️  IMAGES MODE - Advanced Hyperparameter Tuning with Augmentation")
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
        print(f"Classes: {sorted(set(y))}")
        
        # Run advanced Optuna tuning
        tuner = AdvancedOptunaTuner(
            X, y, X, y,  # Using same data for train/test in images mode
            n_trials=self.n_trials
        )
        
        # Run optimization with custom model paths
        results = self._run_optimization_with_custom_paths(
            tuner, 
            self.images_models_dir,
            "images"
        )
        
        self.results["images_mode"] = results
        
        print(f"✅ Images mode completed. Models saved to: {self.images_models_dir}")
        return results
    
    def _run_optimization_with_custom_paths(self, tuner, models_dir, mode_name):
        """
        Run optimization with custom model saving paths.
        """
        # Create study with advanced sampler
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
            
            # Train and evaluate final model
            final_model = tuner._train_final_model(model_name, best_params)
            test_accuracy = final_model.score(tuner.X_test, tuner.y_test)
            
            print(f"{model_name} Test accuracy: {test_accuracy:.4f}")
            
            # Save model with custom path
            model_path = models_dir / f"{model_name.lower()}.joblib"
            joblib.dump(final_model, model_path)
            print(f"Saved {model_name} to {model_path}")
            
            results[model_name] = {
                "best_params": best_params,
                "cv_score": best_score,
                "test_accuracy": test_accuracy,
                "model_path": str(model_path),
                "n_trials": len(study.trials),
                "pruned_trials": sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
            }
            
            # Log to MLflow
            tuner._log_to_mlflow(model_name, best_params, best_score, test_accuracy)
        
        return results
    
    def compare_results(self):
        """
        Compare results between embeddings mode and images mode.
        """
        print("\n📊 COMPREHENSIVE COMPARISON ANALYSIS")
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
                "Images_CV_Score": img_data.get("cv_score", 0),
                "Images_Test_Accuracy": img_data.get("test_accuracy", 0),
                "CV_Score_Difference": img_data.get("cv_score", 0) - emb_data.get("cv_score", 0),
                "Test_Accuracy_Difference": img_data.get("test_accuracy", 0) - emb_data.get("test_accuracy", 0),
                "Best_Mode": "Images" if img_data.get("test_accuracy", 0) > emb_data.get("test_accuracy", 0) else "Embeddings"
            })
        
        # Create DataFrame for analysis
        df = pd.DataFrame(comparison_data)
        
        print("\n📈 DETAILED COMPARISON TABLE:")
        print(df.to_string(index=False, float_format='%.4f'))
        
        # Find best overall models
        best_embeddings = max(embeddings_results.items(), key=lambda x: x[1].get("test_accuracy", 0))
        best_images = max(images_results.items(), key=lambda x: x[1].get("test_accuracy", 0))
        
        print(f"\n🏆 BEST EMBEDDINGS MODE MODEL:")
        print(f"   Algorithm: {best_embeddings[0]}")
        print(f"   Test Accuracy: {best_embeddings[1]['test_accuracy']:.4f}")
        print(f"   CV Score: {best_embeddings[1]['cv_score']:.4f}")
        
        print(f"\n🏆 BEST IMAGES MODE MODEL:")
        print(f"   Algorithm: {best_images[0]}")
        print(f"   Test Accuracy: {best_images[1]['test_accuracy']:.4f}")
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
        
        # Store comparison results
        self.results["comparison"] = {
            "comparison_table": comparison_data,
            "best_embeddings_model": {
                "algorithm": best_embeddings[0],
                "accuracy": best_embeddings[1]['test_accuracy'],
                "cv_score": best_embeddings[1]['cv_score']
            },
            "best_images_model": {
                "algorithm": best_images[0],
                "accuracy": best_images[1]['test_accuracy'],
                "cv_score": best_images[1]['cv_score']
            },
            "overall_winner": {
                "mode": winner,
                "algorithm": winner_model[0],
                "accuracy": winner_model[1]['test_accuracy']
            }
        }
        
        return df
    
    def save_results(self):
        """
        Save all results to JSON and CSV files.
        """
        print(f"\n💾 SAVING RESULTS TO {self.results_dir}")
        print("=" * 60)
        
        # Save detailed results as JSON
        results_file = self.results_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✅ Detailed results saved to: {results_file}")
        
        # Save comparison table as CSV
        if "comparison" in self.results and "comparison_table" in self.results["comparison"]:
            comparison_file = self.results_dir / "comparison_table.csv"
            df = pd.DataFrame(self.results["comparison"]["comparison_table"])
            df.to_csv(comparison_file, index=False)
            print(f"✅ Comparison table saved to: {comparison_file}")
        
        # Save summary report
        summary_file = self.results_dir / "summary_report.txt"
        with open(summary_file, 'w') as f:
            f.write("COMPREHENSIVE FACE RECOGNITION COMPARISON REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {self.results['metadata']['timestamp']}\n")
            f.write(f"Number of trials per algorithm: {self.results['metadata']['n_trials']}\n")
            f.write(f"Algorithms tested: {', '.join(self.results['metadata']['algorithms'])}\n\n")
            
            if "comparison" in self.results:
                comp = self.results["comparison"]
                f.write("OVERALL WINNER:\n")
                f.write(f"  Mode: {comp['overall_winner']['mode']}\n")
                f.write(f"  Algorithm: {comp['overall_winner']['algorithm']}\n")
                f.write(f"  Accuracy: {comp['overall_winner']['accuracy']:.4f}\n\n")
                
                f.write("BEST EMBEDDINGS MODE MODEL:\n")
                f.write(f"  Algorithm: {comp['best_embeddings_model']['algorithm']}\n")
                f.write(f"  Accuracy: {comp['best_embeddings_model']['accuracy']:.4f}\n")
                f.write(f"  CV Score: {comp['best_embeddings_model']['cv_score']:.4f}\n\n")
                
                f.write("BEST IMAGES MODE MODEL:\n")
                f.write(f"  Algorithm: {comp['best_images_model']['algorithm']}\n")
                f.write(f"  Accuracy: {comp['best_images_model']['accuracy']:.4f}\n")
                f.write(f"  CV Score: {comp['best_images_model']['cv_score']:.4f}\n\n")
                
                f.write("MODEL LOCATIONS:\n")
                f.write(f"  Embeddings models: {self.embeddings_models_dir}\n")
                f.write(f"  Images models: {self.images_models_dir}\n")
        
        print(f"✅ Summary report saved to: {summary_file}")
    
    def run_full_comparison(self, emb_dir="data/embeddings", images_dir="data/processed", 
                          n_augment=1, batch_size=16):
        """
        Run the complete comparison between embeddings and images mode.
        """
        start_time = time.time()
        
        print("🚀 STARTING COMPREHENSIVE FACE RECOGNITION COMPARISON")
        print("=" * 60)
        print(f"Results will be saved to: {self.results_dir}")
        print(f"Number of trials per algorithm: {self.n_trials}")
        print(f"Algorithms: SVM, KNN, RandomForest, LogisticRegression")
        print("=" * 60)
        
        # Run embeddings mode
        embeddings_results = self.run_embeddings_mode_comparison(emb_dir)
        
        # Run images mode
        images_results = self.run_images_mode_comparison(images_dir, n_augment, batch_size)
        
        # Compare results
        comparison_df = self.compare_results()
        
        # Save all results
        self.save_results()
        
        total_time = time.time() - start_time
        print(f"\n🎉 COMPARISON COMPLETED IN {total_time:.2f} SECONDS")
        print(f"📁 All results saved to: {self.results_dir}")
        
        return {
            "embeddings_results": embeddings_results,
            "images_results": images_results,
            "comparison": comparison_df,
            "total_time": total_time
        }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Comprehensive comparison of embeddings vs images mode")
    parser.add_argument("--emb_dir", default="data/embeddings", help="Embeddings directory")
    parser.add_argument("--images_dir", default="data/processed", help="Processed images directory")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials per algorithm")
    parser.add_argument("--n_augment", type=int, default=1, help="Number of augmentations for images mode")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for image processing")
    parser.add_argument("--results_dir", default="comparison_results", help="Results directory")
    
    args = parser.parse_args()
    
    # Create comparison system
    comparison = ComprehensiveComparison(
        results_dir=args.results_dir,
        n_trials=args.n_trials
    )
    
    # Run full comparison
    results = comparison.run_full_comparison(
        emb_dir=args.emb_dir,
        images_dir=args.images_dir,
        n_augment=args.n_augment,
        batch_size=args.batch_size
    )
    
    return results

if __name__ == "__main__":
    main()
