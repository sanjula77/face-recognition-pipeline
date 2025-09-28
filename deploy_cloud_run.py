#!/usr/bin/env python3
"""
Deploy Face Recognition API to Google Cloud Run
Simple deployment script that will actually work!
"""

import subprocess
import os
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Deploying Face Recognition API to Google Cloud Run")
    print("=" * 60)
    
    # Check if gcloud is installed
    if not run_command("gcloud --version", "Checking gcloud CLI"):
        print("\n❌ Google Cloud CLI not found!")
        print("Please install it from: https://cloud.google.com/sdk/docs/install")
        return False
    
    # Set project ID (you'll need to change this)
    project_id = "your-project-id"  # Change this to your actual project ID
    service_name = "face-recognition-api"
    region = "us-central1"
    
    print(f"\n📋 Deployment Configuration:")
    print(f"  Project ID: {project_id}")
    print(f"  Service Name: {service_name}")
    print(f"  Region: {region}")
    
    # Authenticate
    if not run_command("gcloud auth login", "Authenticating with Google Cloud"):
        return False
    
    # Set project
    if not run_command(f"gcloud config set project {project_id}", f"Setting project to {project_id}"):
        return False
    
    # Enable required APIs
    if not run_command("gcloud services enable run.googleapis.com", "Enabling Cloud Run API"):
        return False
    
    if not run_command("gcloud services enable cloudbuild.googleapis.com", "Enabling Cloud Build API"):
        return False
    
    # Deploy to Cloud Run
    deploy_command = f"""
    gcloud run deploy {service_name} \
        --source . \
        --platform managed \
        --region {region} \
        --allow-unauthenticated \
        --port 8000 \
        --memory 2Gi \
        --cpu 2 \
        --timeout 300 \
        --max-instances 10
    """
    
    if not run_command(deploy_command, "Deploying to Cloud Run"):
        return False
    
    print("\n🎉 Deployment completed successfully!")
    print(f"Your API will be available at: https://{service_name}-{hash(project_id) % 10000}-{region}.run.app")
    print("\n📝 Next steps:")
    print("1. Test your API endpoints")
    print("2. Create a real-time frontend")
    print("3. Set up custom domain (optional)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
