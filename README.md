# Emotion-Aware Health Companion AI

This project is a multimodal AI-powered web application designed to detect human emotion through facial expression and voice analysis. It also collects user-reported symptoms and provides personalized health advice using OpenAI's GPT-3.5 model.

The application integrates computer vision, audio processing, and natural language understanding to support emotional and physical health awareness.

---

## Features

- Face emotion detection using a CNN-based model
- Voice emotion detection from `.wav` files using audio feature extraction
- Symptom input collection via a text interface
- Personalized health advice generated using the OpenAI API
- PDF report generation summarizing the results
- Secure API usage through a `.env` file (not committed to version control)

---

## Tech Stack

- Python (Streamlit, PyTorch, OpenAI API, Librosa)
- ReportLab for PDF generation
- dotenv for environment variable management
- Git & GitHub for version control

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/vamsi513/Emotion-AI-Companion-Final.git
cd Emotion-AI-Companion-Final
2. Install dependencies
Make sure you are using a virtual environment:
pip install -r requirements.txt
3. Create a .env file
Add your OpenAI API key in the .env file:
OPENAI_API_KEY=your_openai_key_here
This file is excluded from Git tracking by .gitignore.
4. Run the application
streamlit run app.py

Project Structure
Emotion-AI-Companion-Final/
│
├── app.py                    # Main Streamlit app
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and setup
├── .env.example              # Template for environment variables
├── .gitignore                # Files and folders excluded from Git
│
├── models/                   # Pre-trained emotion model
├── utils/                    # Utility scripts (face, voice, PDF)
├── data/, assets/, etc.      # Other resource folders (if applicable)

Security
The .env file should never be uploaded to GitHub. It contains your private API key.

Only .env.example is tracked to show required environment variables.

License
This project is licensed under the MIT License. See LICENSE for details.

Author
Vamsi Krishna sadu
GitHub: vamsi513