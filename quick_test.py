#!/usr/bin/env python3
"""
Quick test for the Face Recognition API
"""

import requests
import time
import subprocess
import sys
from pathlib import Path

def test_api():
    print('🚀 Starting API and testing...')
    
    # Start API in background
    print('Starting API server...')
    process = subprocess.Popen([sys.executable, 'api.py'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    
    # Wait for API to start
    print('Waiting for API to start...')
    time.sleep(8)
    
    try:
        # Test health endpoint
        print('Testing health endpoint...')
        response = requests.get('http://localhost:8000/health', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print('✅ API is working!')
            print('Model loaded:', data['model_loaded'])
            print('Classes:', data['classes'])
            
            # Test face recognition
            test_image = Path('data/test/person_1.jpg')
            if test_image.exists():
                print('Testing face recognition...')
                with open(test_image, 'rb') as f:
                    files = {'file': f}
                    response = requests.post('http://localhost:8000/recognize', files=files, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['success']:
                        print('✅ Face recognition working!')
                        print('Prediction:', data['prediction'])
                        confidence = data['confidence']
                        print('Confidence:', f'{confidence:.1%}')
                        print('Processing time:', f'{data["processing_time"]:.3f}s')
                    else:
                        print('❌ Face recognition failed:', data['message'])
                else:
                    print('❌ Face recognition request failed:', response.status_code)
                    print('Response:', response.text)
            else:
                print('❌ Test image not found at:', test_image)
        else:
            print('❌ Health check failed:', response.status_code)
            print('Response:', response.text)
    
    except Exception as e:
        print('❌ Error:', e)
    finally:
        # Stop the API
        print('Stopping API...')
        process.terminate()
        process.wait()
        print('✅ Test completed!')

if __name__ == "__main__":
    test_api()
