# 🩺 CKD Risk Predictor

A machine learning web app that predicts the risk of **Chronic Kidney Disease (CKD)** based on patient clinical data.

## 🔗 Live Demo
[Try it here](YOUR_STREAMLIT_URL)

## 🧠 Model
- Algorithm: Random Forest Classifier
- Dataset: UCI Kidney Disease Dataset (400 patients, 24 features)
- Accuracy: 100% on test set

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Streamlit
- Pandas / NumPy

## 📊 Features
- Input patient data across 3 categories: blood work, vitals, and clinical history
- Real-time CKD risk probability with a visual gauge
- Key signal indicators (hemoglobin, creatinine, blood urea, etc.)

## 🚀 Run Locally
```bash
git clone https://github.com/sarahgdjl/ckd-predictor.git
cd ckd-predictor
pip install -r requirements.txt
streamlit run app/app.py
```

## ⚠️ Disclaimer
For educational purposes only. Not a substitute for clinical diagnosis.