#!/usr/bin/env python3
"""
Test script for Face Recognition API
"""

import requests
import json
from pathlib import Path

def test_api():
    """Test the face recognition API"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Face Recognition API")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Model loaded: {data['model_loaded']}")
            print(f"   Classes: {data['classes']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Get classes
    print("\n2. Testing classes endpoint...")
    try:
        response = requests.get(f"{base_url}/classes")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Classes endpoint passed")
            print(f"   Found {data['count']} classes: {data['classes']}")
        else:
            print(f"❌ Classes endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Classes endpoint error: {e}")
    
    # Test 3: Face recognition with test image
    print("\n3. Testing face recognition...")
    test_images = list(Path('data/test').glob('*.jpg'))
    
    if not test_images:
        print("❌ No test images found in data/test/")
        return False
    
    test_image = test_images[0]
    print(f"   Using test image: {test_image.name}")
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/recognize", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ Face recognition successful!")
                print(f"   Prediction: {data['prediction']}")
                print(f"   Confidence: {data['confidence']:.1%}")
                print(f"   Processing time: {data['processing_time']:.3f}s")
                print(f"   Faces detected: {data['faces_detected']}")
                print(f"   Top predictions:")
                for pred in data['top_predictions']:
                    print(f"     - {pred['name']}: {pred['confidence']:.1%}")
            else:
                print(f"❌ Face recognition failed: {data['message']}")
        else:
            print(f"❌ Face recognition request failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Face recognition error: {e}")
        return False
    
    # Test 4: Multiple faces recognition
    print("\n4. Testing multiple faces recognition...")
    try:
        with open(test_image, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{base_url}/recognize-multiple", files=files)
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ Multiple faces recognition successful!")
                print(f"   Faces detected: {data['faces_detected']}")
                print(f"   Processing time: {data['processing_time']:.3f}s")
                for result in data['results']:
                    print(f"   Face {result['face_id']}: {result['prediction']} ({result['confidence']:.1%})")
            else:
                print(f"❌ Multiple faces recognition failed: {data['message']}")
        else:
            print(f"❌ Multiple faces recognition request failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Multiple faces recognition error: {e}")
    
    print("\n🎉 API testing completed!")
    return True

if __name__ == "__main__":
    test_api()
