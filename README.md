# 🔐 FaceAuth: Professional AI Biometric Login System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange.svg)](https://scikit-learn.org/)

**FaceAuth** is a high-performance, machine learning-driven authentication system that replaces traditional passwords with secure facial recognition. This project features a premium, Bento-grid inspired user interface, real-time biometric health monitoring, and a comprehensive administrative insights dashboard.

---

## 🌟 Key Features

### 📸 Seamless Face enrollment
- **Automated Data Collection:** High-speed webcam capture with real-time face detection.
- **Structured Storage:** Automatically organizes biometric samples into a secure local data vault.
- **Smart Feedback:** Provides live feedback on lighting levels and face positioning.

### 🧠 Intelligent Authentication
- **KNN-Powered Recognition:** Uses K-Nearest Neighbors for robust and accurate multi-user identification.
- **Confidence Thresholding:** Implements an adjustable security layer to block unauthorized access effectively.
- **Real-Time HUD:** A scanning interface that provides live "Liveness Score" and latency metrics.

### 📊 Advanced Insights Dashboard
- **Login Analytics:** Visualizes login frequency trends over a 30-day window.
- **Outcome Breakdown:** Clear statistics on successful vs. unknown identification attempts.
- **Identity Recognition Table:** Ranked list of the most frequent users by successful authentication.
- **Biometric Health Module:** Real-time system monitoring and integrity checks.

---

## 💪 Strengths & Usefulness

### 1. **User Experience First**
The application isn't just a model—it's a product. The interface uses modern design tokens (Glassmorphism, Bento grids, Material Icons) to provide a premium feel that rivals enterprise software.

### 2. **Modular & Extensible**
The source code is perfectly decoupled. The ML logic (`src/`) is independent of the UI layer (`app/`), allowing for easy swaps of the recognition engine (e.g., swapping KNN for Deep Learning) without breaking the app.

### 3. **Production-Ready Logging**
Every authentication attempt (Success or Unknown) is logged to a structured JSON-Lines file. This creates an auditable trail essential for security monitoring and incident response.

### 4. **Low Barrier to Entry**
Requires no specialized hardware—just a standard webcam. It serves as a perfect template for integrating biometric security into existing Python/Streamlit applications.

---

## 📂 Repository Structure

```text
face-authentication-system/
├── app/                  # UI Application Layer
│   ├── main.py           # Dashboard Entry Point
│   ├── pages/            # Feature Screens (Register, Login, Insights)
│   └── utils.py          # State Management & Logging Helpers
├── src/                  # Core ML & Engine (Logic Layer)
│   ├── data_collection.py # Webcam Management
│   ├── preprocessing.py   # Detection & Cleaning
│   ├── feature_engineering.py # Data Extraction
│   ├── train.py          # Model Fitting (KNN)
│   ├── evaluate.py       # Performance Metrics
│   └── predict.py        # Real-time Inference
├── data/                 # Data Layer
│   └── raw/              # Biometric Sample Directories
├── models/               # Persistence Layer
│   └── face_model.pkl    # Trained Serialized Model
├── config.py             # Global Parameters & Environs
├── requirements.txt      # System Dependencies
└── README.md             # Project Documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- A functional webcam

### Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/y0hannes/face-authentication-system.git
   cd face-authentication-system
   ```

2. **Set up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the System:**
   ```bash
   streamlit run app/main.py
   ```

---

## 🔄 System Workflow

1.  **Enrollment:** Enter your name in the **Register** page. The system captures 10 high-quality face samples and stores them.
2.  **Training:** Click "Synchronize Neural Matrix". The system extracts features and trains the KNN model on the new identities.
3.  **Authentication:** Go to the **Login** page. Activate the camera and watch the system recognize you in real-time.
4.  **Monitoring:** Review the **Insights** page to see your login timestamp, confidence scores, and system benchmarks.

---

## 🔮 Future Roadmap
- [ ] **Embedding Support:** Integrate `FaceNet` or `DeepFace` for vector-based recognition.
- [ ] **Active Liveness:** Add blink detection and head-pose validation.
- [ ] **Cloud Sync:** Integrate Supabase or AWS S3 for remote image storage.

