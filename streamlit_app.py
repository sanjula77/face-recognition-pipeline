#!/usr/bin/env python3
"""
Real-time Face Recognition Streamlit App
Uses trained models for live face detection and recognition
"""

import streamlit as st
import cv2
import numpy as np
import joblib
import insightface
from insightface.app import FaceAnalysis
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import base64

# Page configuration
st.set_page_config(
    page_title="Face Recognition App",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'app_face' not in st.session_state:
    st.session_state.app_face = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = []
if 'recognition_history' not in st.session_state:
    st.session_state.recognition_history = []

@st.cache_resource
def load_models():
    """Load the trained face recognition models"""
    try:
        # Load the trained model
        model_path = Path('production_models/face_recognizer.joblib')
        if not model_path.exists():
            model_path = Path('corrected_comparison_results/embeddings_mode_models/logisticregression.joblib')
        
        if not model_path.exists():
            st.error("❌ No trained model found! Please run the training pipeline first.")
            return None, None, []
        
        model = joblib.load(model_path)
        st.success(f"✅ Model loaded: {type(model).__name__} from {model_path}")
        
        # Initialize InsightFace
        app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app_face.prepare(ctx_id=-1, det_size=(640, 640))
        st.success("✅ InsightFace initialized")
        
        # Load class names from embeddings
        embeddings_dir = Path('data/embeddings')
        if embeddings_dir.exists():
            embedding_files = list(embeddings_dir.glob('*.npy'))
            class_names = sorted(list(set([f.stem.split('_')[0] for f in embedding_files if not f.name.startswith('embeddings_database')])))
            st.success(f"✅ Loaded {len(class_names)} classes: {class_names}")
        else:
            # Fallback class names
            class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']
            st.warning("⚠️ Using fallback class names")
        
        return model, app_face, class_names
        
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, []

def recognize_face(image, model, app_face, class_names):
    """Recognize face in the given image"""
    try:
        # Convert PIL to OpenCV format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert BGR to RGB for InsightFace
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        faces = app_face.get(img_rgb)
        
        if not faces:
            return None, "No face detected"
        
        # Process the first face
        face = faces[0]
        embedding = face.embedding
        
        # Make prediction
        probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
        predicted_name = class_names[prediction_idx]
        
        # Get top 3 predictions
        top_predictions = []
        for i, prob in enumerate(probabilities):
            top_predictions.append({
                "name": class_names[i],
                "confidence": float(prob)
            })
        
        # Sort by confidence
        top_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "prediction": predicted_name,
            "confidence": float(confidence),
            "top_predictions": top_predictions[:3],
            "face_bbox": {
                "x": int(face.bbox[0]),
                "y": int(face.bbox[1]),
                "width": int(face.bbox[2] - face.bbox[0]),
                "height": int(face.bbox[3] - face.bbox[1])
            }
        }, None
        
    except Exception as e:
        return None, f"Recognition error: {e}"

def main():
    # Header
    st.markdown('<h1 class="main-header">👤 Real-time Face Recognition</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Load models button
        if st.button("🔄 Load Models", type="primary"):
            with st.spinner("Loading models..."):
                model, app_face, class_names = load_models()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.app_face = app_face
                    st.session_state.class_names = class_names
                    st.session_state.model_loaded = True
                    st.success("Models loaded successfully!")
                else:
                    st.error("Failed to load models!")
        
        # Model status
        if st.session_state.model_loaded:
            st.markdown('<div class="success-box">✅ Models Ready</div>', unsafe_allow_html=True)
            st.write(f"**Classes:** {', '.join(st.session_state.class_names)}")
        else:
            st.markdown('<div class="error-box">❌ Models Not Loaded</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Recognition settings
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
        show_bbox = st.checkbox("Show Face Bounding Box", value=True)
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Statistics")
        if st.session_state.recognition_history:
            total_recognitions = len(st.session_state.recognition_history)
            successful_recognitions = len([r for r in st.session_state.recognition_history if r.get('success', False)])
            success_rate = (successful_recognitions / total_recognitions) * 100 if total_recognitions > 0 else 0
            
            st.metric("Total Recognitions", total_recognitions)
            st.metric("Success Rate", f"{success_rate:.1f}%")
            
            if st.button("🗑️ Clear History"):
                st.session_state.recognition_history = []
                st.rerun()
    
    # Main content
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the models first using the sidebar button.")
        st.info("💡 Make sure you have trained models in the `production_models/` or `corrected_comparison_results/` directory.")
        return
    
    # Tabs for different modes
    tab1, tab2, tab3, tab4 = st.tabs(["📷 Live Camera", "📁 Upload Image", "📊 Analytics", "ℹ️ About"])
    
    with tab1:
        st.header("🎥 Live Face Recognition")
        st.info("💡 **Real-time face detection** - Just look at the camera! No buttons to click.")
        
        # Initialize live detection state
        if 'live_detection' not in st.session_state:
            st.session_state.live_detection = True  # Auto-start
        if 'live_results' not in st.session_state:
            st.session_state.live_results = []
        if 'last_recognition' not in st.session_state:
            st.session_state.last_recognition = None
        if 'recognition_count' not in st.session_state:
            st.session_state.recognition_count = 0
        
        # Control buttons (minimal)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Restart Detection", type="secondary"):
                st.session_state.live_detection = True
                st.session_state.recognition_count = 0
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.live_results = []
                st.session_state.recognition_count = 0
                st.rerun()
        
        # Live detection status
        if st.session_state.live_detection:
            st.success("🎥 **Live detection is ACTIVE** - Look at your camera!")
        else:
            st.error("⏹️ Live detection is STOPPED")
        
        # Results display area
        results_placeholder = st.empty()
        
        # Live camera input (continuous)
        if st.session_state.live_detection:
            # Use camera input for live stream
            live_camera = st.camera_input(
                "Look at the camera for face recognition", 
                key="live_camera",
                help="Position your face clearly in the camera view"
            )
            
            if live_camera is not None:
                # Process the live frame
                image = Image.open(live_camera)
                
                # Recognize face
                result, error = recognize_face(image, st.session_state.model, st.session_state.app_face, st.session_state.class_names)
                
                # Update recognition count
                st.session_state.recognition_count += 1
                
                if result:
                    # Check if this is a new recognition (avoid spam)
                    current_prediction = f"{result['prediction']}_{result['confidence']:.2f}"
                    if st.session_state.last_recognition != current_prediction:
                        st.session_state.last_recognition = current_prediction
                        
                        if result['confidence'] >= confidence_threshold:
                            # High confidence - recognized
                            with results_placeholder.container():
                                st.markdown(f"""
                                <div class="success-box">
                                    <h2>✅ RECOGNIZED: {result['prediction']}</h2>
                                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                                    <p><strong>Status:</strong> 🎯 High Confidence Match</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add to live results
                            st.session_state.live_results.append({
                                'timestamp': time.time(),
                                'prediction': result['prediction'],
                                'confidence': result['confidence'],
                                'status': 'recognized'
                            })
                            
                            # Add to history
                            st.session_state.recognition_history.append({
                                'timestamp': time.time(),
                                'prediction': result['prediction'],
                                'confidence': result['confidence'],
                                'success': True
                            })
                        
                        elif result['confidence'] >= 0.5:
                            # Medium confidence
                            with results_placeholder.container():
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>❓ {result['prediction']}</h3>
                                    <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                                    <p><strong>Status:</strong> ⚠️ Medium Confidence</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        else:
                            # Low confidence
                            with results_placeholder.container():
                                st.markdown(f"""
                                <div class="error-box">
                                    <h3>❌ Unknown Person</h3>
                                    <p><strong>Best Match:</strong> {result['prediction']} ({result['confidence']:.1%})</p>
                                    <p><strong>Status:</strong> 🚫 Low Confidence</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                else:
                    # No face detected
                    with results_placeholder.container():
                        st.markdown(f"""
                        <div class="error-box">
                            <h3>👁️ No Face Detected</h3>
                            <p>Please position your face clearly in the camera view.</p>
                            <p><strong>Tips:</strong> Ensure good lighting and face is centered</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Live statistics
        if st.session_state.live_results:
            st.subheader("📊 Live Recognition Stats")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Scans", st.session_state.recognition_count)
            
            with col2:
                recognized_count = len([r for r in st.session_state.live_results if r.get('status') == 'recognized'])
                st.metric("Recognitions", recognized_count)
            
            with col3:
                if st.session_state.live_results:
                    latest = st.session_state.live_results[-1]
                    st.metric("Latest Confidence", f"{latest['confidence']:.1%}")
            
            # Recent recognitions
            st.subheader("📋 Recent Recognitions")
            recent_live = st.session_state.live_results[-10:]  # Last 10 results
            for i, result in enumerate(reversed(recent_live)):
                status_icon = "✅" if result.get('status') == 'recognized' else "❓"
                st.write(f"{status_icon} **{result['prediction']}** - {result['confidence']:.1%} ({time.strftime('%H:%M:%S', time.localtime(result['timestamp']))})")
    
    with tab2:
        st.header("📁 Upload Image for Recognition")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📁 Uploaded Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("🔍 Recognition Results")
                
                # Recognize face
                with st.spinner("Recognizing face..."):
                    result, error = recognize_face(image, st.session_state.model, st.session_state.app_face, st.session_state.class_names)
                
                if result:
                    # Display results
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>👤 {result['prediction']}</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top predictions
                    st.subheader("🏆 Top Predictions")
                    for i, pred in enumerate(result['top_predictions']):
                        st.write(f"{i+1}. **{pred['name']}** - {pred['confidence']:.2%}")
                    
                    # Add to history
                    st.session_state.recognition_history.append({
                        'timestamp': time.time(),
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'success': result['confidence'] >= confidence_threshold
                    })
                    
                else:
                    st.error(f"❌ {error}")
    
    with tab3:
        st.header("📊 Recognition Analytics")
        
        if st.session_state.recognition_history:
            # Convert to DataFrame
            df = pd.DataFrame(st.session_state.recognition_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Recognitions", len(df))
            
            with col2:
                success_rate = (df['success'].sum() / len(df)) * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            with col3:
                avg_confidence = df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            with col4:
                most_recognized = df['prediction'].mode().iloc[0] if not df.empty else "N/A"
                st.metric("Most Recognized", most_recognized)
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Recognition Count by Person")
                person_counts = df['prediction'].value_counts()
                st.bar_chart(person_counts)
            
            with col2:
                st.subheader("📊 Confidence Distribution")
                fig, ax = plt.subplots()
                ax.hist(df['confidence'], bins=20, alpha=0.7, color='skyblue')
                ax.set_xlabel('Confidence')
                ax.set_ylabel('Frequency')
                ax.set_title('Confidence Distribution')
                st.pyplot(fig)
            
            # Recent recognitions table
            st.subheader("📋 Recent Recognitions")
            recent_df = df.tail(10)[['timestamp', 'prediction', 'confidence', 'success']].copy()
            recent_df['confidence'] = recent_df['confidence'].apply(lambda x: f"{x:.1%}")
            recent_df['success'] = recent_df['success'].apply(lambda x: "✅" if x else "❌")
            st.dataframe(recent_df, use_container_width=True)
            
        else:
            st.info("📊 No recognition data available yet. Start recognizing faces to see analytics!")
    
    with tab4:
        st.header("ℹ️ About This App")
        
        st.markdown("""
        ### 🎯 **Face Recognition App**
        
        This Streamlit app provides real-time face recognition using trained machine learning models.
        
        ### ✨ **Features**
        - 📷 **Live Camera Recognition** - Real-time face detection from webcam
        - 📁 **Image Upload** - Upload images for face recognition
        - 📊 **Analytics Dashboard** - View recognition statistics and trends
        - 🎛️ **Customizable Settings** - Adjust confidence thresholds
        - 📈 **Performance Metrics** - Track recognition accuracy
        
        ### 🔧 **Technical Details**
        - **Face Detection**: InsightFace (RetinaFace)
        - **Face Recognition**: Trained ML models (Logistic Regression, SVM, etc.)
        - **Embedding Model**: ArcFace (512-dimensional)
        - **Framework**: Streamlit + OpenCV + scikit-learn
        
        ### 📚 **How It Works**
        1. **Face Detection** - Locate faces in the image
        2. **Feature Extraction** - Generate face embeddings
        3. **Classification** - Predict person identity using trained model
        4. **Confidence Scoring** - Provide prediction confidence
        
        ### 🚀 **Getting Started**
        1. Click "Load Models" in the sidebar
        2. Choose "Live Camera" or "Upload Image"
        3. Start recognizing faces!
        
        ### 📁 **Model Files**
        - `production_models/face_recognizer.joblib` - Main trained model
        - `data/embeddings/` - Face embeddings database
        - `corrected_comparison_results/` - Model comparison results
        """)

if __name__ == "__main__":
    main()
