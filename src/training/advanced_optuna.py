#!/usr/bin/env python3
"""
Advanced Optuna hyperparameter tuning for face recognition classifiers.
Includes multi-objective optimization, pruning, and advanced samplers.
"""

import optuna
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

class AdvancedOptunaTuner:
    """
    Advanced hyperparameter tuning using Optuna with multiple algorithms and objectives.
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, n_trials=50):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.n_trials = n_trials
        self.results = {}
        
    def svm_objective(self, trial):
        """SVM hyperparameter optimization with advanced parameters."""
        # Advanced SVM parameters
        C = trial.suggest_float("C", 1e-4, 1e4, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
        
        if kernel == "rbf":
            gamma = trial.suggest_float("gamma", 1e-5, 1e2, log=True)
            clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        elif kernel == "poly":
            degree = trial.suggest_int("degree", 2, 5)
            gamma = trial.suggest_float("gamma", 1e-5, 1e2, log=True)
            clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, probability=True, random_state=42)
        else:  # linear
            clf = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        
        # Cross-validation with pruning
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(clf, self.X_train, self.y_train, cv=cv, scoring='accuracy', n_jobs=1)
        
        # Pruning: stop if performance is too low
        if scores.mean() < 0.1:
            raise optuna.TrialPruned()
            
        return scores.mean()
    
    def knn_objective(self, trial):
        """KNN hyperparameter optimization with advanced parameters."""
        n_samples = len(self.X_train)
        max_k = min(20, n_samples - 1)
        
        k = trial.suggest_int("k", 1, max_k)
        metric = trial.suggest_categorical("metric", ["cosine", "euclidean", "manhattan"])
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        algorithm = trial.suggest_categorical("algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
        
        # Fix algorithm-metric compatibility
        if metric == "cosine" and algorithm in ["kd_tree", "ball_tree"]:
            algorithm = "brute"  # cosine only works with brute force
        
        clf = KNeighborsClassifier(
            n_neighbors=k, 
            metric=metric, 
            weights=weights, 
            algorithm=algorithm
        )
        
        try:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=cv, scoring='accuracy', n_jobs=1)
            
            if scores.mean() < 0.1:
                raise optuna.TrialPruned()
                
            return scores.mean()
        except Exception as e:
            # Skip failed trials
            raise optuna.TrialPruned()
    
    def random_forest_objective(self, trial):
        """Random Forest hyperparameter optimization."""
        n_estimators = trial.suggest_int("n_estimators", 50, 500)
        max_depth = trial.suggest_int("max_depth", 3, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42
        )
        
        try:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=cv, scoring='accuracy', n_jobs=1)
            
            if scores.mean() < 0.1:
                raise optuna.TrialPruned()
                
            return scores.mean()
        except Exception as e:
            # Skip failed trials
            raise optuna.TrialPruned()
    
    def logistic_regression_objective(self, trial):
        """Logistic Regression hyperparameter optimization."""
        C = trial.suggest_float("C", 1e-4, 1e4, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
        
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)
            clf = LogisticRegression(C=C, penalty=penalty, solver=solver, l1_ratio=l1_ratio, random_state=42)
        else:
            clf = LogisticRegression(C=C, penalty=penalty, solver=solver, random_state=42)
        
        try:
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(clf, self.X_train, self.y_train, cv=cv, scoring='accuracy', n_jobs=1)
            
            if scores.mean() < 0.1:
                raise optuna.TrialPruned()
                
            return scores.mean()
        except Exception as e:
            # Skip failed trials
            raise optuna.TrialPruned()
    
    def multi_objective_optimization(self, trial):
        """Multi-objective optimization (accuracy + precision + recall)."""
        # Use SVM as example
        C = trial.suggest_float("C", 1e-4, 1e4, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        
        if kernel == "rbf":
            gamma = trial.suggest_float("gamma", 1e-5, 1e2, log=True)
            clf = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=42)
        else:
            clf = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        # Multiple objectives
        accuracy_scores = cross_val_score(clf, self.X_train, self.y_train, cv=cv, scoring='accuracy', n_jobs=1)
        precision_scores = cross_val_score(clf, self.X_train, self.y_train, cv=cv, scoring='precision_macro', n_jobs=1)
        recall_scores = cross_val_score(clf, self.X_train, self.y_train, cv=cv, scoring='recall_macro', n_jobs=1)
        
        return accuracy_scores.mean(), precision_scores.mean(), recall_scores.mean()
    
    def optimize_all_models(self, use_pruning=True, use_multi_objective=False):
        """Optimize all models with advanced Optuna features."""
        
        # Create study with advanced sampler
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=10,  # Random trials before Bayesian optimization
            n_ei_candidates=24,   # Number of candidates for expected improvement
            seed=42
        )
        
        if use_pruning:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=3,  # Don't prune first 3 trials
                n_warmup_steps=1,    # Wait 1 step before pruning
                interval_steps=1     # Check pruning every step
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        models_to_optimize = {
            "SVM": self.svm_objective,
            "KNN": self.knn_objective,
            "RandomForest": self.random_forest_objective,
            "LogisticRegression": self.logistic_regression_objective
        }
        
        if use_multi_objective:
            # Multi-objective optimization
            study = optuna.create_study(
                directions=["maximize", "maximize", "maximize"],
                sampler=sampler,
                pruner=pruner
            )
            study.optimize(self.multi_objective_optimization, n_trials=self.n_trials)
            
            print("Multi-objective optimization results:")
            print(f"Number of Pareto-optimal solutions: {len(study.best_trials)}")
            for i, trial in enumerate(study.best_trials):
                print(f"Solution {i+1}: Accuracy={trial.values[0]:.4f}, Precision={trial.values[1]:.4f}, Recall={trial.values[2]:.4f}")
            
            return study
        
        # Single-objective optimization for each model
        for model_name, objective_func in models_to_optimize.items():
            print(f"\n🔍 Optimizing {model_name}...")
            
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
            final_model = self._train_final_model(model_name, best_params)
            test_accuracy = final_model.score(self.X_test, self.y_test)
            
            print(f"{model_name} Test accuracy: {test_accuracy:.4f}")
            
            # Save model
            model_path = f"models/advanced_{model_name.lower()}.joblib"
            joblib.dump(final_model, model_path)
            print(f"Saved {model_name} to {model_path}")
            
            self.results[model_name] = {
                "best_params": best_params,
                "cv_score": best_score,
                "test_accuracy": test_accuracy,
                "model_path": model_path
            }
            
            # Log to MLflow
            self._log_to_mlflow(model_name, best_params, best_score, test_accuracy)
        
        return self.results
    
    def _train_final_model(self, model_name, params):
        """Train final model with best parameters."""
        if model_name == "SVM":
            if params["kernel"] == "rbf":
                model = SVC(C=params["C"], kernel=params["kernel"], gamma=params["gamma"], probability=True, random_state=42)
            elif params["kernel"] == "poly":
                model = SVC(C=params["C"], kernel=params["kernel"], degree=params["degree"], gamma=params["gamma"], probability=True, random_state=42)
            else:
                model = SVC(C=params["C"], kernel=params["kernel"], probability=True, random_state=42)
            # Fit the model
            model.fit(self.X_train, self.y_train)
            return model
        
        elif model_name == "KNN":
            # Fix algorithm-metric compatibility for final model
            algorithm = params["algorithm"]
            if params["metric"] == "cosine" and algorithm in ["kd_tree", "ball_tree"]:
                algorithm = "brute"  # cosine only works with brute force
            
            model = KNeighborsClassifier(
                n_neighbors=params["k"],
                metric=params["metric"],
                weights=params["weights"],
                algorithm=algorithm
            )
            model.fit(self.X_train, self.y_train)
            return model
        
        elif model_name == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=42
            )
            model.fit(self.X_train, self.y_train)
            return model
        
        elif model_name == "LogisticRegression":
            if params["penalty"] == "elasticnet":
                model = LogisticRegression(
                    C=params["C"],
                    penalty=params["penalty"],
                    solver=params["solver"],
                    l1_ratio=params["l1_ratio"],
                    random_state=42
                )
            else:
                model = LogisticRegression(
                    C=params["C"],
                    penalty=params["penalty"],
                    solver=params["solver"],
                    random_state=42
                )
            model.fit(self.X_train, self.y_train)
            return model
    
    def _log_to_mlflow(self, model_name, params, cv_score, test_accuracy):
        """Log results to MLflow."""
        mlflow.set_tracking_uri("file:///mlflow.db")
        with mlflow.start_run(run_name=f"advanced_{model_name.lower()}_tuning"):
            mlflow.log_param("model", model_name)
            mlflow.log_param("n_trials", self.n_trials)
            
            # Log all hyperparameters
            for param_name, param_value in params.items():
                mlflow.log_param(f"{param_name}", param_value)
            
            mlflow.log_metric("cv_accuracy", cv_score)
            mlflow.log_metric("test_accuracy", test_accuracy)
    
    def plot_optimization_history(self, study):
        """Plot optimization history (requires optuna.visualization)."""
        try:
            import optuna.visualization as vis
            import plotly
            
            # Optimization history
            fig1 = vis.plot_optimization_history(study)
            fig1.write_html("optimization_history.html")
            
            # Parameter importance
            fig2 = vis.plot_param_importances(study)
            fig2.write_html("param_importances.html")
            
            # Parallel coordinate plot
            fig3 = vis.plot_parallel_coordinate(study)
            fig3.write_html("parallel_coordinate.html")
            
            print("📊 Optimization plots saved:")
            print("  - optimization_history.html")
            print("  - param_importances.html") 
            print("  - parallel_coordinate.html")
            
        except ImportError:
            print("📊 Install optuna[visualization] to generate plots: pip install optuna[visualization]")

def main():
    """Example usage of advanced Optuna tuning."""
    print("🚀 Advanced Optuna Hyperparameter Tuning")
    print("=" * 50)
    
    # This would be called from the main training script
    # with actual data loaded
    print("To use this advanced tuner, integrate it into your training pipeline.")
    print("Example:")
    print("  tuner = AdvancedOptunaTuner(X_train, y_train, X_test, y_test, n_trials=100)")
    print("  results = tuner.optimize_all_models(use_pruning=True)")

if __name__ == "__main__":
    main()
