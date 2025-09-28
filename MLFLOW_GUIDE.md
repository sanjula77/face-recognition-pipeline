# 🚀 MLflow Guide for Face Recognition Project

## ✅ **MLflow is Already Set Up!**

Your project already has MLflow configured and running with:
- **MLflow version:** 3.4.0
- **Tracking URI:** `file:///C:/Users/ASUS/Desktop/face/face-recognition-project/mlruns`
- **Experiments:** 1 experiment with 8 runs
- **Web UI:** Running on http://localhost:5000

## 🌐 **Access MLflow Web Dashboard**

### **Step 1: Open Web Browser**
1. Open your web browser (Chrome, Firefox, Edge, etc.)
2. Go to: **http://localhost:5000**
3. You'll see the MLflow dashboard

### **Step 2: Explore Your Experiments**
- Click on **"Default"** experiment
- You'll see **8 runs** from your face recognition pipeline
- Each run represents a different step in your pipeline

## 📊 **What You Can See in MLflow UI**

### **📈 Experiment Overview**
- **Run names** and timestamps
- **Status** (Finished, Running, Failed)
- **Duration** of each run
- **Parameters** used in each run
- **Metrics** (accuracy, F1-score, processing time)

### **🔍 Individual Run Details**
Click on any run to see:
- **Parameters:** Model settings, hyperparameters
- **Metrics:** Performance scores, processing times
- **Artifacts:** Sample faces, embeddings, trained models
- **Tags:** Run metadata and descriptions

### **📊 Model Comparison**
- **Compare different algorithms:** SVM vs KNN vs RandomForest vs LogisticRegression
- **View performance metrics** side by side
- **See which hyperparameters** work best
- **Identify the best performing model**

## 🎯 **How to Use MLflow with Your Pipeline**

### **When You Run the Pipeline:**
```bash
python run_complete_pipeline.py
```

**MLflow automatically tracks:**
1. **Preprocessing runs** - face detection parameters and results
2. **Embedding extraction** - model settings and performance
3. **Model training** - hyperparameters and accuracy metrics
4. **Artifacts** - sample faces, embeddings, trained models

### **New Runs Will Appear:**
- Each time you run the pipeline, new runs will appear in MLflow
- You can compare **old vs new** performance
- Track improvements as you **add more people** to your dataset

## 🔧 **MLflow Commands**

### **Start MLflow UI:**
```bash
mlflow ui
```

### **View Experiments Programmatically:**
```python
import mlflow

# List all experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name}, ID: {exp.experiment_id}")

# Get latest run
runs = mlflow.search_runs(experiment_ids=[0])
latest_run = runs.iloc[0]
print(f"Latest run: {latest_run['run_id']}")
print(f"Accuracy: {latest_run['metrics.accuracy']}")
```

## 📈 **Benefits for Your Project**

### **1. Track Improvements**
- **Before adding new people:** Note current performance
- **After adding new people:** Compare new performance
- **Identify if scaling up** improves or hurts performance

### **2. Model Selection**
- **Compare all 4 algorithms** (SVM, KNN, RandomForest, LogisticRegression)
- **See which performs best** on your specific dataset
- **Track hyperparameter optimization** results

### **3. Reproducibility**
- **Exact same results** every time you run the pipeline
- **Track all parameters** used in each run
- **Easy to reproduce** outstanding results

### **4. Production Deployment**
- **Identify best model** from MLflow UI
- **Download production-ready model**
- **Deploy with confidence**

## 🎯 **Next Steps**

1. **Open http://localhost:5000** in your browser
2. **Explore your existing 8 runs**
3. **Compare model performances**
4. **When you add new people** to your dataset, run the pipeline again
5. **Use MLflow to track improvements** and select the best model

## 💡 **Pro Tips**

- **Bookmark http://localhost:5000** for easy access
- **Check MLflow before and after** adding new people
- **Use the comparison feature** to see which models work best
- **Download artifacts** (sample faces, embeddings) for analysis
- **Track processing times** to optimize your pipeline

---

**MLflow is now your experiment tracking companion!** 🚀
