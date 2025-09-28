#!/usr/bin/env python3
"""
Simple launcher for the Streamlit Face Recognition App
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting Face Recognition Streamlit App...")
    print("=" * 50)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
    except ImportError:
        print("❌ Streamlit not found! Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed!")
    
    # Check if required files exist
    required_files = [
        "streamlit_app.py",
        "production_models/face_recognizer.joblib",
        "data/embeddings/"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Warning: Some files are missing:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Make sure you have trained models before running the app.")
        print("   You can still run the app, but it may not work properly.")
    
    print("\n🌐 Starting Streamlit app...")
    print("📱 The app will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("\n⏹️  Press Ctrl+C to stop the app")
    print("=" * 50)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"\n❌ Error running app: {e}")

if __name__ == "__main__":
    main()
