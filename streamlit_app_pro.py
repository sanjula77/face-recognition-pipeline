#!/usr/bin/env python3
"""
Professional Face Recognition Streamlit App
Production-ready with modern UI, advanced metrics, and comprehensive features
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
import seaborn as sns
from PIL import Image
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

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
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #f7fafc;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
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
    
    /* Progress Bars */
    .progress-container {
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
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
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
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
    """Load the trained face recognition models with enhanced error handling"""
    try:
        logger.info("Loading face recognition models...")
        
        # Load the trained model
        model_path = Path('production_models/face_recognizer.joblib')
        if not model_path.exists():
            model_path = Path('corrected_comparison_results/embeddings_mode_models/logisticregression.joblib')
        
        if not model_path.exists():
            st.error("❌ No trained model found! Please run the training pipeline first.")
            return None, None, [], {}
        
        # Load model with timing
        start_time = time.time()
        model = joblib.load(model_path)
        model_load_time = time.time() - start_time
        
        # Initialize InsightFace
        start_time = time.time()
        app_face = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app_face.prepare(ctx_id=-1, det_size=(640, 640))
        insightface_init_time = time.time() - start_time
        
        # Load class names from embeddings
        embeddings_dir = Path('data/embeddings')
        if embeddings_dir.exists():
            embedding_files = list(embeddings_dir.glob('*.npy'))
            class_names = sorted(list(set([f.stem.split('_')[0] for f in embedding_files if not f.name.startswith('embeddings_database')])))
        else:
            class_names = ['ameesha', 'gihan', 'keshan', 'lakshan', 'oshanda', 'pasindu', 'ravishan']
        
        # Calculate model metrics
        model_metrics = {
            'model_type': type(model).__name__,
            'model_path': str(model_path),
            'class_count': len(class_names),
            'model_load_time': model_load_time,
            'insightface_init_time': insightface_init_time,
            'total_load_time': model_load_time + insightface_init_time,
            'classes': class_names
        }
        
        logger.info(f"Models loaded successfully in {model_metrics['total_load_time']:.2f}s")
        return model, app_face, class_names, model_metrics
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"❌ Error loading models: {e}")
        import traceback
        st.error(f"Full error: {traceback.format_exc()}")
        return None, None, [], {}

def recognize_face(image, model, app_face, class_names, debug=False):
    """Enhanced face recognition with performance metrics"""
    start_time = time.time()
    
    try:
        if debug:
            st.write(f"🔍 Debug: Processing image of size {image.size if hasattr(image, 'size') else 'unknown'}")
        
        # Convert PIL to OpenCV format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert BGR to RGB for InsightFace
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if debug:
            st.write(f"🔍 Debug: Image converted, shape: {img_rgb.shape}")
        
        # Detect faces
        faces = app_face.get(img_rgb)
        
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
        probabilities = model.predict_proba(embedding.reshape(1, -1))[0]
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
            import traceback
            st.error(f"Full error: {traceback.format_exc()}")
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
            st.markdown(f"""
            <div class="info-card">
                <h4>Model Information</h4>
                <p><strong>Type:</strong> {metrics.get('model_type', 'Unknown')}</p>
                <p><strong>Classes:</strong> {metrics.get('class_count', 0)}</p>
                <p><strong>Load Time:</strong> {metrics.get('total_load_time', 0):.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="info-card">
                <h4>System Status</h4>
                <p><span class="status-indicator status-online"></span><strong>Status:</strong> Online</p>
                <p><strong>Last Update:</strong> {datetime.now().strftime('%H:%M:%S')}</p>
                <p><strong>Uptime:</strong> {time.time() - st.session_state.get('start_time', time.time()):.0f}s</p>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Face Recognition Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; font-size: 1.2rem; margin-bottom: 3rem;">Professional-grade face recognition with advanced analytics and monitoring</p>', unsafe_allow_html=True)
    
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
        
        st.session_state.settings['show_bbox'] = st.checkbox(
            "Show Bounding Box", st.session_state.settings['show_bbox']
        )
        
        st.session_state.settings['debug_mode'] = st.checkbox(
            "Debug Mode", st.session_state.settings['debug_mode']
        )
        
        st.session_state.settings['auto_save'] = st.checkbox(
            "Auto Save Results", st.session_state.settings['auto_save']
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
            <p>Make sure you have trained models in the <code>production_models/</code> or <code>corrected_comparison_results/</code> directory.</p>
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
        st.markdown('<h2 class="sub-header">🎥 Live Face Recognition</h2>', unsafe_allow_html=True)
        
        # Live detection interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📹 Camera Feed")
            st.markdown("**Instructions:** Look directly at the camera. Face detection will happen automatically when you appear in the frame.")
            
            live_camera = st.camera_input(
                "🎥 Live Face Detection", 
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
                    
                else:
                    st.error(f"❌ {error}")
    
    with tab3:
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
                    "Load Time": f"{metrics.get('total_load_time', 0):.2f}s"
                })
            
            with col2:
                st.markdown("### 📊 Class Distribution")
                if metrics.get('classes'):
                    class_df = pd.DataFrame({
                        'Class': metrics['classes'],
                        'Count': [1] * len(metrics['classes'])  # Placeholder
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
            st.session_state.settings['show_bbox'] = st.checkbox(
                "Show Face Bounding Box", st.session_state.settings['show_bbox']
            )
            st.session_state.settings['debug_mode'] = st.checkbox(
                "Enable Debug Mode", st.session_state.settings['debug_mode']
            )
        
        with col2:
            st.markdown("### 💾 Data Management")
            st.session_state.settings['auto_save'] = st.checkbox(
                "Auto Save Results", st.session_state.settings['auto_save']
            )
            
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
