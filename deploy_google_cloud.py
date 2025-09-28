#!/usr/bin/env python3
"""
Google Cloud Run Deployment Script
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 Google Cloud Run Deployment")
    print("=" * 50)
    
    # Check if gcloud is installed
    if not run_command("gcloud --version", "Checking Google Cloud CLI"):
        print("\n❌ Google Cloud CLI not found!")
        print("Please install it from: https://cloud.google.com/sdk/docs/install")
        return False
    
    # Authenticate
    if not run_command("gcloud auth login", "Authenticating with Google Cloud"):
        return False
    
    # Set project (you'll need to create one)
    project_id = input("\nEnter your Google Cloud Project ID (or press Enter to create one): ").strip()
    
    if not project_id:
        project_id = "face-recognition-api"
        print(f"Creating project: {project_id}")
        if not run_command(f"gcloud projects create {project_id}", "Creating Google Cloud project"):
            return False
    
    if not run_command(f"gcloud config set project {project_id}", "Setting project"):
        return False
    
    # Enable required APIs
    if not run_command("gcloud services enable run.googleapis.com", "Enabling Cloud Run API"):
        return False
    
    if not run_command("gcloud services enable cloudbuild.googleapis.com", "Enabling Cloud Build API"):
        return False
    
    # Deploy to Cloud Run
    print("\n🚀 Deploying to Google Cloud Run...")
    deploy_command = f"""
    gcloud run deploy face-recognition-api \
        --source . \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 1 \
        --timeout 300 \
        --max-instances 10
    """
    
    if run_command(deploy_command, "Deploying to Cloud Run"):
        print("\n🎉 Deployment successful!")
        print("Your API will be available at:")
        print(f"https://face-recognition-api-{project_id}.uc.r.appspot.com")
        print("\nTest it with:")
        print("curl https://face-recognition-api-{project_id}.uc.r.appspot.com/health")
        return True
    else:
        return False

if __name__ == "__main__":
    main()
