#!/usr/bin/env python3
"""
Start Face Recognition API locally with public access
"""

import subprocess
import sys
import time
import requests
import threading

def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting Face Recognition API...")
    try:
        # Start the API server
        process = subprocess.Popen([
            sys.executable, "api.py"
        ])
        
        # Wait for API to start
        time.sleep(8)
        
        # Test if API is running
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API started successfully on http://localhost:8000")
                return process
            else:
                print("❌ API not responding properly")
                return None
        except:
            print("❌ API not responding")
            return None
            
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        return None

def show_instructions():
    """Show instructions for accessing the API"""
    print("\n" + "="*60)
    print("🎉 FACE RECOGNITION API IS RUNNING!")
    print("="*60)
    print("\n📋 API Endpoints:")
    print("   • Health Check: http://localhost:8000/health")
    print("   • Face Recognition: http://localhost:8000/recognize")
    print("   • API Documentation: http://localhost:8000/docs")
    print("\n🧪 Test Commands:")
    print("   curl http://localhost:8000/health")
    print("   curl -X POST http://localhost:8000/recognize -F 'file=@data/test/person_1.jpg'")
    print("\n🌐 For Public Access:")
    print("   1. Install ngrok: https://ngrok.com/download")
    print("   2. Run: ngrok http 8000")
    print("   3. Use the ngrok URL to access your API from anywhere")
    print("\n🛑 Press Ctrl+C to stop the server")
    print("="*60)

def main():
    print("🚀 Starting Face Recognition API Server")
    print("="*50)
    
    # Start API
    api_process = start_api()
    if not api_process:
        print("❌ Failed to start API")
        return
    
    # Show instructions
    show_instructions()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping API server...")
        api_process.terminate()
        api_process.wait()
        print("✅ API server stopped")

if __name__ == "__main__":
    main()
