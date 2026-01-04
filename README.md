# CareWatch – AI-Powered Health Assistant

## Overview
CareWatch is an AI-powered health assistant designed to support users with personalized dietary guidance and medical insights. The system allows users to log meals using text or voice input, upload medical reports for analysis, and receive health-aware recommendations based on extracted data.

The project focuses on combining machine learning, OCR, and NLP techniques with a user-friendly interface to support preventive healthcare decision-making.

---

## Key Features
- Text and voice-based food input
- Nutritional analysis and health guidance
- OCR-based medical report processing
- Personalized diet recommendations
- Daily health journaling
- Streamlit-based interactive user interface
- Dockerized application for deployment

---

## Project Structure
- `app.py` — Main Streamlit application
- `medical_report_core.py` — Medical report OCR and analysis logic
- `logger_setup.py` — Centralized logging configuration
- `datasets/` — Sample and reference datasets
- `CareWatch_Use_Case_implementation.ipynb` — Use case demonstration notebook
- `Dockerfile` — Container configuration
- `requirements.txt` — Project dependencies

---

## Technologies Used
- Python
- Streamlit
- Pandas & NumPy
- OCR (Tesseract)
- Natural Language Processing (NLP)
- Docker
- Logging & configuration management

---

## How to Run the Project

### Local Setup
```bash
pip install -r requirements.txt
streamlit run app.py

### Docker Setup
docker build -t carewatch .
docker run -p 8501:8501 carewatch

### Use Case
CareWatch is intended as a decision-support tool for:
- Individuals monitoring dietary habits
- Users with chronic health conditions
- Preventive healthcare and wellness tracking

### Notes
- This project is for educational and research purposes.
- No real patient data is included in this repository.

### Author
Abdul Bari Mohammed
