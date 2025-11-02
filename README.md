# ❤️ Heart Disease Prediction Web App

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/M-F-Tushar/Heart-Disease-Prediction-Web-App/actions/workflows/ci.yml/badge.svg)](https://github.com/M-F-Tushar/Heart-Disease-Prediction-Web-App/actions/workflows/ci.yml)

A web application that predicts heart disease risk using machine learning. Built with Streamlit and scikit-learn.

---

## 🩺 What is this app?
This app uses a machine learning model trained on real patient data to estimate the risk of heart disease. Enter patient information, and the app will predict the risk and show the probability.

## 🌐 Try it Online
Deployed on [Streamlit Community Cloud](https://m-f-tushar-heart-disease-prediction-web-app-app-hwx2ii.streamlit.app/).

---

## 🌟 Features

- **Easy-to-use web interface** - No technical knowledge required
- **Real-time predictions** with probability scores
- **Interactive input forms** with helpful tooltips
- **Medical disclaimer** for responsible use
- **Automatic model creation** if not present
- **Input validation** to ensure data quality
- **Responsive design** that works on all devices

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/M-F-Tushar/Heart-Disease-Prediction-Web-App.git
   cd Heart-Disease-Prediction-Web-App
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:8501
   ```

---

## 📊 Model Performance

- **Algorithm:** Logistic Regression
- **Accuracy:** ~85% on test data
- **Dataset:** UCI Heart Disease (303 patients)
- **Features:** 13 medical attributes

---

## 🚀 How to Use
1. Enter the patient's medical information in the form fields.
2. Click **Predict Heart Disease Risk**.
3. View the prediction and probability.

---

## ⚙️ Deployment

### Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Update the CI badge in README.md with your repository URL (optional)
3. Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in with GitHub
4. Click **New app**, select your forked repo, and set the main file to `app.py`
5. Click **Deploy**. Your app will be live in a few minutes!

### Required Files for Deployment
- `app.py` - Main Streamlit application
- `heart-disease.csv` - Dataset for model training
- `requirements.txt` - Python dependencies
- `.gitignore` - Files to exclude from version control

---

## 📋 About the Data
- **Source:** UCI Heart Disease dataset
- **Samples:** 303 patients
- **Features:** Age, sex, chest pain, blood pressure, cholesterol, blood sugar, ECG, heart rate, angina, ST depression, slope, vessels, thalassemia

---

## ⚕️ Disclaimer
This tool is for **educational purposes only** and is **not** a diagnostic tool. It was trained on a small dataset and should not be used for real medical decisions. Always consult qualified healthcare providers.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to get started.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

*Built with ❤️ using Streamlit and scikit-learn*
