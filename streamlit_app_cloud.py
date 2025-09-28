#!/usr/bin/env python3
"""
Streamlit Cloud Compatible Face Recognition App
Optimized for deployment without OpenCV dependencies
"""

import streamlit as st
import numpy as np
import joblib
import time
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import OpenCV, fallback if not available
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("⚠️ OpenCV not available - using fallback mode")

# Try to import InsightFace, fallback if not available
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    st.warning("⚠️ InsightFace not available - using mock mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Face Recognition Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Face Recognition Pro\nProfessional-grade face recognition system with advanced analytics and monitoring."
    }
)

# Professional CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    /* Card Styles */
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border: 1px solid #9ae6b4;
        color: #22543d;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #f6e05e;
        color: #92400e;
    }
    
    .error-card {
        background: linear-gradient(135deg, #fed7d7 0%, #feb2b2 100%);
        border: 1px solid #fc8181;
        color: #742a2a;
    }
    
    .info-card {
        background: linear-gradient(135deg, #ebf8ff 0%, #bee3f8 100%);
        border: 1px solid #63b3ed;
        color: #2a4365;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #48bb78; }
    .status-offline { background-color: #f56565; }
    .status-warning { background-color: #ed8936; }
    
    /* Custom Metrics */
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2d3748;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'model_loaded': False,
        'app_face': None,
        'model': None,
        'class_names': [],
        'recognition_history': [],
        'model_metrics': {},
        'system_status': 'offline',
        'last_update': None,
        'total_recognitions': 0,
        'successful_recognitions': 0,
        'avg_confidence': 0.0,
        'processing_time': 0.0,
        'live_detection': True,
        'live_results': [],
        'last_recognition': None,
        'recognition_count': 0,
        'model_performance': {},
        'settings': {
            'confidence_threshold': 0.7,
            'show_bbox': True,
            'auto_save': True,
            'debug_mode': False,
            'theme': 'light'
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

@st.cache_resource
def load_models():
    """Load the trained face recognition models with cloud compatibility"""
    try:
        logger.info("Loading face recognition models...")
        
        # Check if we're in cloud mode
        if not OPENCV_AVAILABLE or not INSIGHTFACE_AVAILABLE:
            st.warning("⚠️ Running in cloud compatibility mode")
            return create_mock_models()
        
        # Load the trained model
        model_path = 'production_models/face_recognizer.joblib'
        try:
            model = joblib.load(model_path)
        except:
            model_path = 'corrected_comparison_results/embeddings_mode_models/logisticregression.joblib'
            model = joblib.load(model_path)
        
        # Initialize InsightFace
        app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app_face.prepare(ctx_id=-1, det_size=(640, 640))
        
        # Load class names
        class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']
        
        # Calculate model metrics
        model_metrics = {
            'model_type': type(model).__name__,
            'model_path': model_path,
            'class_count': len(class_names),
            'classes': class_names,
            'cloud_mode': False
        }
        
        logger.info("Models loaded successfully")
        return model, app_face, class_names, model_metrics
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.warning("⚠️ Using mock models for demonstration")
        return create_mock_models()

def create_mock_models():
    """Create mock models for cloud deployment"""
    class MockModel:
        def predict_proba(self, X):
            # Return random probabilities for demo - ensure proper shape
            np.random.seed(42)
            # X should be 2D array, return probabilities for 7 classes
            if X.ndim == 1:
                X = X.reshape(1, -1)
            batch_size = X.shape[0]
            # Return 2D array with shape (batch_size, num_classes)
            return np.random.dirichlet(np.ones(7), size=batch_size)
    
    class MockFaceAnalysis:
        def get(self, img):
            # Return mock face data
            class MockFace:
                def __init__(self):
                    self.embedding = np.random.randn(512)
                    self.bbox = [100, 100, 200, 200]
            return [MockFace()]
    
    model = MockModel()
    app_face = MockFaceAnalysis()
    class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']
    
    model_metrics = {
        'model_type': 'MockModel',
        'model_path': 'mock',
        'class_count': len(class_names),
        'classes': class_names,
        'cloud_mode': True
    }
    
    return model, app_face, class_names, model_metrics

def recognize_face(image, model, app_face, class_names, debug=False):
    """Enhanced face recognition with cloud compatibility"""
    start_time = time.time()
    
    try:
        if debug:
            st.write(f"🔍 Debug: Processing image of size {image.size if hasattr(image, 'size') else 'unknown'}")
        
        # Convert PIL to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        if debug:
            st.write(f"🔍 Debug: Image converted, shape: {img_array.shape}")
        
        # Detect faces
        faces = app_face.get(img_array)
        
        if debug:
            st.write(f"🔍 Debug: Found {len(faces)} faces")
        
        if not faces:
            return None, "No face detected", 0.0
        
        # Process the first face
        face = faces[0]
        embedding = face.embedding
        
        if debug:
            st.write(f"🔍 Debug: Face embedding shape: {embedding.shape}")
        
        # Make prediction
        probabilities = model.predict_proba(embedding.reshape(1, -1))
        # Handle both 1D and 2D probability arrays
        if probabilities.ndim == 2:
            probabilities = probabilities[0]  # Take first row if 2D
        prediction_idx = np.argmax(probabilities)
        confidence = probabilities[prediction_idx]
        predicted_name = class_names[prediction_idx]
        
        if debug:
            st.write(f"🔍 Debug: Prediction: {predicted_name}, Confidence: {confidence:.3f}")
        
        # Get top 3 predictions
        top_predictions = []
        for i, prob in enumerate(probabilities):
            top_predictions.append({
                "name": class_names[i],
                "confidence": float(prob)
            })
        
        # Sort by confidence
        top_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        processing_time = time.time() - start_time
        
        result = {
            "prediction": predicted_name,
            "confidence": float(confidence),
            "top_predictions": top_predictions[:3],
            "face_bbox": {
                "x": int(face.bbox[0]),
                "y": int(face.bbox[1]),
                "width": int(face.bbox[2] - face.bbox[0]),
                "height": int(face.bbox[3] - face.bbox[1])
            },
            "processing_time": processing_time,
            "timestamp": time.time()
        }
        
        return result, None, processing_time
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Recognition error: {e}")
        if debug:
            st.error(f"❌ Recognition error: {e}")
        return None, f"Recognition error: {e}", processing_time

def create_performance_dashboard():
    """Create comprehensive performance dashboard"""
    st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.total_recognitions}</div>
            <div class="metric-label">Total Recognitions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        success_rate = (st.session_state.successful_recognitions / max(st.session_state.total_recognitions, 1)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{success_rate:.1f}%</div>
            <div class="metric-label">Success Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.avg_confidence:.1%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{st.session_state.processing_time:.2f}s</div>
            <div class="metric-label">Avg Processing Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts
    if st.session_state.recognition_history:
        col1, col2 = st.columns(2)
        
        with col1:
            # Recognition trends
            df = pd.DataFrame(st.session_state.recognition_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['timestamp'].dt.hour
            
            hourly_counts = df.groupby('hour').size().reset_index(name='count')
            
            fig = px.line(hourly_counts, x='hour', y='count', 
                         title='Recognition Activity by Hour',
                         color_discrete_sequence=['#667eea'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence distribution
            fig = px.histogram(df, x='confidence', nbins=20, 
                              title='Confidence Distribution',
                              color_discrete_sequence=['#764ba2'])
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

def create_model_metrics_section():
    """Create detailed model metrics section"""
    st.markdown('<h2 class="sub-header">🤖 Model Performance</h2>', unsafe_allow_html=True)
    
    if st.session_state.model_metrics:
        metrics = st.session_state.model_metrics
        
        col1, col2 = st.columns(2)
        
        with col1:
            mode_text = "Cloud Demo Mode" if metrics.get('cloud_mode', False) else "Production Mode"
            st.markdown(f"""
            <div class="info-card">
                <h4>Model Information</h4>
                <p><strong>Type:</strong> {metrics.get('model_type', 'Unknown')}</p>
                <p><strong>Classes:</strong> {metrics.get('class_count', 0)}</p>
                <p><strong>Mode:</strong> {mode_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h4>System Status</h4>
                <p><span class="status-indicator status-online"></span><strong>Status:</strong> Online</p>
                <p><strong>Last Update:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                <p><strong>Environment:</strong> Streamlit Cloud</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Face Recognition Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; font-size: 1.2rem; margin-bottom: 3rem;">Professional-grade face recognition with advanced analytics and monitoring</p>', unsafe_allow_html=True)
    
    # Cloud compatibility notice
    if not OPENCV_AVAILABLE or not INSIGHTFACE_AVAILABLE:
        st.markdown("""
        <div class="warning-card">
            <h3>☁️ Cloud Compatibility Mode</h3>
            <p>This app is running in cloud compatibility mode. Some features may be limited.</p>
            <p><strong>Available:</strong> Image upload, mock recognition, analytics dashboard</p>
            <p><strong>Limited:</strong> Real face detection, camera input</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # System status
        status_color = "success-card" if st.session_state.model_loaded else "error-card"
        status_text = "🟢 Online" if st.session_state.model_loaded else "🔴 Offline"
        
        st.markdown(f"""
        <div class="{status_color}">
            <h4>System Status</h4>
            <p>{status_text}</p>
            <p><strong>Models:</strong> {'Loaded' if st.session_state.model_loaded else 'Not Loaded'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Load models button
        if st.button("🔄 Load Models", type="primary", use_container_width=True):
            with st.spinner("Loading models..."):
                model, app_face, class_names, model_metrics = load_models()
                if model is not None:
                    st.session_state.model = model
                    st.session_state.app_face = app_face
                    st.session_state.class_names = class_names
                    st.session_state.model_metrics = model_metrics
                    st.session_state.model_loaded = True
                    st.session_state.system_status = 'online'
                    st.session_state.start_time = time.time()
                    st.success("✅ Models loaded successfully!")
                    st.rerun()
                else:
                    st.error("❌ Failed to load models!")
        
        st.divider()
        
        # Settings
        st.markdown("### ⚙️ Settings")
        
        st.session_state.settings['confidence_threshold'] = st.slider(
            "Confidence Threshold", 0.0, 1.0, st.session_state.settings['confidence_threshold'], 0.05
        )
        
        st.session_state.settings['debug_mode'] = st.checkbox(
            "Debug Mode", st.session_state.settings['debug_mode']
        )
        
        st.divider()
        
        # Quick stats
        if st.session_state.recognition_history:
            st.markdown("### 📈 Quick Stats")
            st.metric("Total Scans", st.session_state.total_recognitions)
            st.metric("Success Rate", f"{(st.session_state.successful_recognitions/max(st.session_state.total_recognitions,1)*100):.1f}%")
            st.metric("Avg Confidence", f"{st.session_state.avg_confidence:.1%}")
            
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.recognition_history = []
                st.session_state.total_recognitions = 0
                st.session_state.successful_recognitions = 0
                st.rerun()
    
    # Main content
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="warning-card">
            <h3>⚠️ Models Not Loaded</h3>
            <p>Please load the face recognition models using the sidebar button.</p>
            <p>In cloud mode, this will load mock models for demonstration.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎥 Live Detection", 
        "📁 Upload Image", 
        "📊 Analytics", 
        "🤖 Model Metrics", 
        "⚙️ Settings"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">🎥 Live Face Detection</h2>', unsafe_allow_html=True)
        
        # Cloud compatibility notice
        st.markdown("""
        <div class="info-card">
            <h3>☁️ Cloud Live Detection</h3>
            <p>In cloud mode, we use camera input for demonstration. Real face detection is simulated.</p>
            <p><strong>Features:</strong> Camera input, mock recognition, real-time results</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Live camera input
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📹 Camera Feed")
            live_camera = st.camera_input(
                "🎥 Live Face Detection - Look at camera", 
                key="live_camera",
                help="Position your face clearly in the camera view for automatic recognition"
            )
        
        with col2:
            st.markdown("### 🎯 Detection Status")
            if live_camera is not None:
                st.markdown("""
                <div class="success-card">
                    <h4>✅ Camera Active</h4>
                    <p>👁️ Looking for faces...</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Process the live frame
                image = Image.open(live_camera)
                result, error, processing_time = recognize_face(
                    image, st.session_state.model, st.session_state.app_face, 
                    st.session_state.class_names, st.session_state.settings['debug_mode']
                )
                
                # Update metrics
                st.session_state.total_recognitions += 1
                st.session_state.processing_time = processing_time
                
                if result:
                    # Check if this is a new recognition
                    current_prediction = f"{result['prediction']}_{result['confidence']:.2f}"
                    if st.session_state.last_recognition != current_prediction:
                        st.session_state.last_recognition = current_prediction
                        
                        if result['confidence'] >= st.session_state.settings['confidence_threshold']:
                            # High confidence - recognized
                            st.markdown(f"""
                            <div class="success-card">
                                <h3>✅ RECOGNIZED</h3>
                                <h2>{result['prediction']}</h2>
                                <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                                <p><strong>Processing Time:</strong> {result['processing_time']:.3f}s</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.session_state.successful_recognitions += 1
                            
                            # Add to history
                            st.session_state.recognition_history.append({
                                'timestamp': result['timestamp'],
                                'prediction': result['prediction'],
                                'confidence': result['confidence'],
                                'success': True,
                                'processing_time': result['processing_time']
                            })
                        
                        elif result['confidence'] >= 0.5:
                            # Medium confidence
                            st.markdown(f"""
                            <div class="warning-card">
                                <h3>❓ {result['prediction']}</h3>
                                <p><strong>Confidence:</strong> {result['confidence']:.1%}</p>
                                <p><strong>Status:</strong> ⚠️ Medium Confidence</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        else:
                            # Low confidence
                            st.markdown(f"""
                            <div class="error-card">
                                <h3>❌ Unknown Person</h3>
                                <p><strong>Best Match:</strong> {result['prediction']} ({result['confidence']:.1%})</p>
                                <p><strong>Status:</strong> 🚫 Low Confidence</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:
                    st.markdown(f"""
                    <div class="error-card">
                        <h3>👁️ No Face Detected</h3>
                        <p>Please position your face clearly in the camera view.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    <h4>⚠️ Camera not active</h4>
                    <p>📷 Click camera to start</p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">📁 Upload Image for Recognition</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📁 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.markdown("### 🔍 Recognition Results")
                
                with st.spinner("Recognizing face..."):
                    result, error, processing_time = recognize_face(
                        image, st.session_state.model, st.session_state.app_face, 
                        st.session_state.class_names, st.session_state.settings['debug_mode']
                    )
                
                if result:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>👤 {result['prediction']}</h3>
                        <p><strong>Confidence:</strong> {result['confidence']:.2%}</p>
                        <p><strong>Processing Time:</strong> {result['processing_time']:.3f}s</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Top predictions
                    st.markdown("### 🏆 Top Predictions")
                    for i, pred in enumerate(result['top_predictions']):
                        st.write(f"{i+1}. **{pred['name']}** - {pred['confidence']:.2%}")
                    
                    # Add to history
                    st.session_state.recognition_history.append({
                        'timestamp': result['timestamp'],
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'success': result['confidence'] >= st.session_state.settings['confidence_threshold'],
                        'processing_time': result['processing_time']
                    })
                    
                    # Update metrics
                    st.session_state.total_recognitions += 1
                    if result['confidence'] >= st.session_state.settings['confidence_threshold']:
                        st.session_state.successful_recognitions += 1
                    st.session_state.avg_confidence = np.mean([r['confidence'] for r in st.session_state.recognition_history])
                    st.session_state.processing_time = processing_time
                    
                else:
                    st.error(f"❌ {error}")
    
    with tab2:
        create_performance_dashboard()
        
        # Recent recognitions table
        if st.session_state.recognition_history:
            st.markdown('<h3 class="sub-header">📋 Recent Recognitions</h3>', unsafe_allow_html=True)
            
            df = pd.DataFrame(st.session_state.recognition_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df['confidence'] = df['confidence'].apply(lambda x: f"{x:.1%}")
            df['success'] = df['success'].apply(lambda x: "✅" if x else "❌")
            df['processing_time'] = df['processing_time'].apply(lambda x: f"{x:.3f}s")
            
            st.dataframe(
                df[['timestamp', 'prediction', 'confidence', 'success', 'processing_time']].tail(20),
                use_container_width=True
            )
    
    with tab4:
        create_model_metrics_section()
        
        # Model comparison
        if st.session_state.model_metrics:
            st.markdown('<h3 class="sub-header">🔍 Model Details</h3>', unsafe_allow_html=True)
            
            metrics = st.session_state.model_metrics
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "Model Type": metrics.get('model_type', 'Unknown'),
                    "Model Path": metrics.get('model_path', 'Unknown'),
                    "Number of Classes": metrics.get('class_count', 0),
                    "Cloud Mode": metrics.get('cloud_mode', False)
                })
            
            with col2:
                st.markdown("### 📊 Class Distribution")
                if metrics.get('classes'):
                    class_df = pd.DataFrame({
                        'Class': metrics['classes'],
                        'Count': [1] * len(metrics['classes'])
                    })
                    fig = px.pie(class_df, values='Count', names='Class', 
                                title='Training Classes')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown('<h2 class="sub-header">⚙️ System Settings</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎛️ Recognition Settings")
            st.session_state.settings['confidence_threshold'] = st.slider(
                "Confidence Threshold", 0.0, 1.0, st.session_state.settings['confidence_threshold'], 0.05
            )
            st.session_state.settings['debug_mode'] = st.checkbox(
                "Enable Debug Mode", st.session_state.settings['debug_mode']
            )
        
        with col2:
            st.markdown("### 💾 Data Management")
            if st.button("📥 Export Results"):
                if st.session_state.recognition_history:
                    df = pd.DataFrame(st.session_state.recognition_history)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"face_recognition_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No data to export")
            
            if st.button("🗑️ Clear All Data"):
                st.session_state.recognition_history = []
                st.session_state.total_recognitions = 0
                st.session_state.successful_recognitions = 0
                st.success("Data cleared!")

if __name__ == "__main__":
    main()
