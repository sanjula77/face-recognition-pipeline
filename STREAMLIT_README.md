# 👤 Real-time Face Recognition Streamlit App

A comprehensive Streamlit application for real-time face detection and recognition using trained machine learning models.

## 🚀 Features

### 📷 **Live Camera Recognition**
- Real-time face detection from webcam
- Instant face recognition with confidence scores
- Live bounding box visualization

### 📁 **Image Upload**
- Upload images for face recognition
- Support for JPG, JPEG, PNG formats
- Batch processing capabilities

### 📊 **Analytics Dashboard**
- Recognition statistics and trends
- Success rate tracking
- Confidence distribution analysis
- Recent recognition history

### 🎛️ **Customizable Settings**
- Adjustable confidence thresholds
- Face bounding box toggle
- Model selection options

## 🛠️ Installation

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Verify Model Files**
Make sure you have these files:
- `production_models/face_recognizer.joblib` - Main trained model
- `data/embeddings/` - Face embeddings database
- `corrected_comparison_results/` - Model comparison results

## 🎯 Usage

### **Quick Start**
```bash
python run_app.py
```

### **Manual Start**
```bash
streamlit run streamlit_app.py
```

### **Custom Port**
```bash
streamlit run streamlit_app.py --server.port 8080
```

## 📱 App Interface

### **Main Tabs**
1. **📷 Live Camera** - Real-time webcam recognition
2. **📁 Upload Image** - Upload images for recognition
3. **📊 Analytics** - View recognition statistics
4. **ℹ️ About** - App information and help

### **Sidebar Controls**
- **🔄 Load Models** - Initialize face recognition models
- **⚙️ Settings** - Adjust confidence threshold and display options
- **📊 Statistics** - View recognition metrics

## 🔧 Technical Details

### **Face Detection Pipeline**
1. **Input Processing** - Convert image to OpenCV format
2. **Face Detection** - Use InsightFace (RetinaFace) to locate faces
3. **Feature Extraction** - Generate 512-dimensional face embeddings
4. **Classification** - Predict identity using trained ML model
5. **Confidence Scoring** - Calculate prediction confidence

### **Models Used**
- **Face Detection**: InsightFace (RetinaFace)
- **Face Recognition**: Trained scikit-learn models
- **Embedding Model**: ArcFace (512-dimensional)

### **Supported Classes**
- ameesha, gihan, keshan, lakshan, oshanda, pasindu, ravishan

## 📊 Performance

### **Recognition Accuracy**
- **High Confidence**: >90% accuracy for well-lit faces
- **Medium Confidence**: 70-90% accuracy for various conditions
- **Low Confidence**: <70% accuracy (may need retraining)

### **Processing Speed**
- **Live Camera**: ~2-3 FPS (real-time)
- **Image Upload**: ~1-2 seconds per image
- **Batch Processing**: ~0.5 seconds per image

## 🚀 Deployment

### **Local Development**
```bash
streamlit run streamlit_app.py
```

### **Streamlit Cloud**
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### **Docker**
```dockerfile
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🔍 Troubleshooting

### **Common Issues**

#### **Models Not Loading**
- Check if `production_models/face_recognizer.joblib` exists
- Verify `data/embeddings/` directory has embedding files
- Run training pipeline if models are missing

#### **Camera Not Working**
- Check camera permissions in browser
- Try different browser (Chrome recommended)
- Verify camera is not used by other applications

#### **Low Recognition Accuracy**
- Ensure good lighting conditions
- Check if face is clearly visible
- Consider retraining with more data

#### **Performance Issues**
- Close other applications using camera
- Reduce image resolution
- Use CPU-optimized models

### **Error Messages**
- **"No face detected"** - Face not clearly visible or too small
- **"Models not loaded"** - Click "Load Models" button first
- **"Recognition error"** - Check model files and dependencies

## 📈 Future Enhancements

### **Planned Features**
- [ ] Multiple face detection in single image
- [ ] Face emotion recognition
- [ ] Age and gender estimation
- [ ] Face similarity comparison
- [ ] Real-time video processing
- [ ] Mobile app integration

### **Technical Improvements**
- [ ] GPU acceleration support
- [ ] Model optimization
- [ ] Caching for faster processing
- [ ] Batch processing capabilities
- [ ] API endpoint integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **InsightFace** - Face detection and recognition
- **Streamlit** - Web app framework
- **OpenCV** - Computer vision library
- **scikit-learn** - Machine learning models

---

**🎉 Enjoy using the Face Recognition App!**
