# src/embeddings/extractor.py
import os
import sys

# Fix OpenMP library conflict on Windows
# This is needed when multiple libraries (NumPy, SciPy, InsightFace) use OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import time
from pathlib import Path
import numpy as np
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import settings
from src.embeddings.utils import batch_extract_embeddings, create_embedding_database

def init_arcface_model(model_name: str = 'buffalo_l', ctx_id: int = 0):
    """
    Initialize ArcFace model for embedding extraction.
    
    Args:
        model_name: Model name (buffalo_l, buffalo_s, etc.)
        ctx_id: Device ID (0 for GPU, -1 for CPU)
        
    Returns:
        Initialized ArcFace model
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        
        print(f"Initializing ArcFace model: {model_name}")
        
        # Initialize FaceAnalysis with detection and recognition modules
        app = FaceAnalysis(name=model_name, allowed_modules=['detection', 'recognition'])
        app.prepare(ctx_id=ctx_id)
        
        print("ArcFace model initialized successfully")
        return app
        
    except Exception as e:
        print(f"Failed to initialize ArcFace model: {e}")
        raise

def extract_embeddings_pipeline(processed_dir: str, embeddings_dir: str, 
                               model_name: str = 'buffalo_l', ctx_id: int = 0):
    """
    Complete pipeline for extracting face embeddings.
    
    Args:
        processed_dir: Directory containing processed face .npy files
        embeddings_dir: Directory to save embeddings
        model_name: ArcFace model name
        ctx_id: Device ID
    """
    # Initialize MLflow tracking if available
    use_mlflow = HAS_MLFLOW
    mlflow_context = None
    
    if use_mlflow:
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            run_name = f"embedding_extraction_{int(time.time())}"
            mlflow_context = mlflow.start_run(run_name=run_name)
        except Exception:
            use_mlflow = False
            mlflow_context = None
    
    # Log parameters
    if use_mlflow:
        try:
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("ctx_id", ctx_id)
            mlflow.log_param("processed_dir", processed_dir)
            mlflow.log_param("embeddings_dir", embeddings_dir)
        except Exception:
            pass
    
    try:
        # Initialize ArcFace model
        model = init_arcface_model(model_name, ctx_id)
        
        # Extract embeddings
        print("\nStarting embedding extraction...")
        start_time = time.time()
        
        stats = batch_extract_embeddings(processed_dir, embeddings_dir, model)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Log metrics
        if use_mlflow:
            try:
                mlflow.log_metric("total_faces", stats['total_faces'])
                mlflow.log_metric("successful_extractions", stats['successful_extractions'])
                mlflow.log_metric("failed_extractions", stats['failed_extractions'])
                mlflow.log_metric("processing_time", processing_time)
                mlflow.log_metric("extraction_rate", stats['successful_extractions'] / max(stats['total_faces'], 1))
            except Exception:
                pass
        
        # Create embedding database
        print("\nCreating embedding database...")
        database_file = Path(embeddings_dir) / "embeddings_database.npz"
        database = create_embedding_database(embeddings_dir, str(database_file))
        
        if database and use_mlflow:
            try:
                mlflow.log_metric("total_embeddings", database['total_embeddings'])
                mlflow.log_metric("embedding_dimension", database['embedding_dim'])
                mlflow.log_artifact(str(database_file), artifact_path="embedding_database")
            except Exception as e:
                print(f"Warning: Could not log database artifact to MLflow: {e}")
        
        # Log sample embeddings as artifacts
        if stats['embeddings_saved'] and use_mlflow:
            sample_embedding = stats['embeddings_saved'][0]
            try:
                mlflow.log_artifact(sample_embedding, artifact_path="sample_embedding")
            except Exception as e:
                print(f"Warning: Could not log sample embedding artifact to MLflow: {e}")
        
        # Print results
        print(f"\nEmbedding extraction completed!")
        print(f"   Total faces processed: {stats['total_faces']}")
        print(f"   Successful extractions: {stats['successful_extractions']}")
        print(f"   Failed extractions: {stats['failed_extractions']}")
        print(f"   Processing time: {processing_time:.2f} seconds")
        print(f"   Extraction rate: {stats['successful_extractions']/max(stats['total_faces'], 1):.1%}")
        
        if database:
            print(f"   Embedding dimension: {database['embedding_dim']}")
            print(f"   Database saved: {database_file}")
        
        return stats, database
        
    except Exception as e:
        print(f"Embedding extraction failed: {e}")
        if use_mlflow:
            try:
                mlflow.log_param("error", str(e))
            except Exception:
                pass
        raise
    finally:
        # Close MLflow run if it was opened
        if mlflow_context:
            try:
                mlflow.end_run()
            except Exception:
                pass

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Extract face embeddings using ArcFace")
    parser.add_argument("--processed_dir", default=settings.PROCESSED_DIR,
                       help="Directory containing processed face .npy files")
    parser.add_argument("--embeddings_dir", default="data/embeddings",
                       help="Directory to save embeddings")
    parser.add_argument("--model", default="buffalo_l",
                       help="ArcFace model name (buffalo_l, buffalo_s)")
    parser.add_argument("--ctx_id", type=int, default=0,
                       help="Device ID (0 for GPU, -1 for CPU)")
    
    args = parser.parse_args()
    
    print("Face Embedding Extraction Pipeline")
    print("=" * 50)
    print(f"Processed directory: {args.processed_dir}")
    print(f"Embeddings directory: {args.embeddings_dir}")
    print(f"Model: {args.model}")
    print(f"Device: {'GPU' if args.ctx_id == 0 else 'CPU'}")
    print("=" * 50)
    
    # Run extraction pipeline
    stats, database = extract_embeddings_pipeline(
        processed_dir=args.processed_dir,
        embeddings_dir=args.embeddings_dir,
        model_name=args.model,
        ctx_id=args.ctx_id
    )
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()