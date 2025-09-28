#!/usr/bin/env python3
"""
Start Face Recognition API in GitHub Codespaces
"""

import subprocess
import sys
import os
import time

def install_ngrok():
    """Install ngrok for public access"""
    print("🔧 Installing ngrok...")
    try:
        # Download and install ngrok
        subprocess.run([
            "curl", "-s", "https://ngrok-agent.s3.amazonaws.com/ngrok.asc", 
            "|", "sudo", "tee", "/etc/apt/trusted.gpg.d/ngrok.asc", ">/dev/null"
        ], shell=True, check=True)
        
        subprocess.run([
            "echo", "'deb https://ngrok-agent.s3.amazonaws.com buster main'", 
            "|", "sudo", "tee", "/etc/apt/sources.list.d/ngrok.list"
        ], shell=True, check=True)
        
        subprocess.run(["sudo", "apt", "update"], check=True)
        subprocess.run(["sudo", "apt", "install", "ngrok"], check=True)
        
        print("✅ ngrok installed successfully")
        return True
    except Exception as e:
        print(f"❌ Error installing ngrok: {e}")
        return False

def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting Face Recognition API...")
    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, "api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for API to start
        time.sleep(5)
        print("✅ API started on localhost:8000")
        return process
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        return None

def start_ngrok(port=8000):
    """Start ngrok tunnel"""
    print(f"🌐 Starting ngrok tunnel on port {port}...")
    try:
        # Start ngrok
        ngrok_process = subprocess.Popen([
            "ngrok", "http", str(port)
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(3)
        print("✅ ngrok tunnel started")
        print("🌐 Your API will be available at the ngrok URL")
        print("📋 Check the ngrok dashboard at: http://localhost:4040")
        return ngrok_process
    except Exception as e:
        print(f"❌ Error starting ngrok: {e}")
        return None

def main():
    print("🚀 GitHub Codespaces Face Recognition API Setup")
    print("=" * 60)
    
    # Check if we're in Codespaces
    if os.getenv('CODESPACES'):
        print("✅ Running in GitHub Codespaces")
    else:
        print("⚠️  Not running in Codespaces - this script is designed for Codespaces")
    
    # Install ngrok
    if not install_ngrok():
        print("❌ Failed to install ngrok")
        return
    
    # Start API
    api_process = start_api()
    if not api_process:
        print("❌ Failed to start API")
        return
    
    # Start ngrok
    ngrok_process = start_ngrok()
    if not ngrok_process:
        print("❌ Failed to start ngrok")
        api_process.terminate()
        return
    
    print("\n🎉 Setup complete!")
    print("📋 Your face recognition API is now running")
    print("🌐 Access it via the ngrok URL shown above")
    print("🛑 Press Ctrl+C to stop")
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        api_process.terminate()
        ngrok_process.terminate()
        print("✅ Services stopped")

if __name__ == "__main__":
    main()
