#!/usr/bin/env python3
"""
Professional Face Recognition App Launcher
Enhanced with better error handling and system checks
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'streamlit', 'opencv-python', 'numpy', 'pandas', 
        'matplotlib', 'seaborn', 'pillow', 'plotly', 'insightface'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_model_files():
    """Check if required model files exist"""
    model_paths = [
        'production_models/face_recognizer.joblib',
        'corrected_comparison_results/embeddings_mode_models/logisticregression.joblib'
    ]
    
    existing_models = []
    for path in model_paths:
        if Path(path).exists():
            existing_models.append(path)
    
    return existing_models

def print_banner():
    """Print professional banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║              🎯 FACE RECOGNITION PRO 🎯                     ║
    ║                                                              ║
    ║              Professional Face Recognition System            ║
    ║              with Advanced Analytics & Monitoring            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    print_banner()
    
    print("🔍 System Check...")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"✅ Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required!")
        return
    
    # Check dependencies
    print("\n📦 Checking Dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("🔧 Installing missing packages...")
        
        for package in missing:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return
    else:
        print("✅ All dependencies satisfied")
    
    # Check model files
    print("\n🤖 Checking Model Files...")
    existing_models = check_model_files()
    
    if existing_models:
        print(f"✅ Found models: {', '.join(existing_models)}")
    else:
        print("⚠️  Warning: No trained models found!")
        print("   Make sure you have trained models in:")
        print("   - production_models/face_recognizer.joblib")
        print("   - corrected_comparison_results/embeddings_mode_models/")
        print("\n   You can still run the app, but it may not work properly.")
    
    # Check data directories
    print("\n📁 Checking Data Directories...")
    data_dirs = ['data/embeddings', 'data/processed', 'data/raw']
    
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path} exists")
        else:
            print(f"⚠️  {dir_path} not found")
    
    print("\n" + "=" * 60)
    print("🚀 Starting Face Recognition Pro...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the app")
    print("=" * 60)
    
    # Start the app
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app_pro.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light",
            "--theme.primaryColor", "#667eea",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f7fafc",
            "--theme.textColor", "#2d3748"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")

if __name__ == "__main__":
    main()
