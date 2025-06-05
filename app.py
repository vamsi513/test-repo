from utils.pdf_utils import generate_pdf
from utils.nlp_utils import generate_health_advice
from utils.face_utils import load_emotion_model, predict_emotion
from utils.speech_utils import load_voice_model, predict_voice_emotion

import streamlit as st

st.set_page_config(
    page_title="Emotion-Aware Health Companion",
    page_icon="🧠",
    layout="centered"
)


# Title and Description
st.title("🧠 Emotion-Aware Health Companion")
st.markdown("""
Welcome to your personal AI health assistant.  
This tool analyzes your **symptoms**, **facial expression**, and **voice tone** to generate preliminary **health advice**.

💡 *Note: This is not a substitute for professional medical consultation.*
""")
st.markdown("---")


st.header("📝 Describe Your Symptoms")
symptoms = st.text_area("What symptoms are you experiencing today?", height=150)

st.markdown("---")

st.header("📷 Upload Facial Image")
face_image = st.file_uploader("Upload a clear face photo", type=["jpg", "jpeg", "png"])

st.markdown("---")

st.header("🎤 Upload Voice Sample")
voice_file = st.file_uploader("Upload a short voice file (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])



# Submit Button
# Submit Button
if st.button("🔍 Analyze"):
    st.success("Inputs received! (Processing now...)")
    
    # Face Emotion Detection
    if face_image is not None:
        model = load_emotion_model()
        predicted_emotion = predict_emotion(model, face_image)
        st.subheader("🧠 Detected Emotion from Face:")
        st.info(f"**{predicted_emotion}**")
    else:
        st.warning("No face image uploaded yet!")

    # ➡️ Now add Voice Emotion Detection immediately below!

    if voice_file is not None:
        voice_model = load_voice_model()
        predicted_voice_emotion = predict_voice_emotion(voice_model, voice_file)
        st.subheader("🎤 Detected Emotion from Voice:")
        st.info(f"**{predicted_voice_emotion}**")
    else:
        st.warning("No voice file uploaded yet!")
    
        # Symptom Text Analysis and Health Advice
    if symptoms.strip() != "":
        advice = generate_health_advice(symptoms)
        st.subheader("💬 Health Advice Based on Your Symptoms:")
        st.success(advice)
    else:
        st.warning("No symptoms entered yet!")
        # Offer PDF Download (only if symptoms were entered)
# Store results in session so they persist during rerun
    st.session_state['face_emotion'] = predicted_emotion
    st.session_state['voice_emotion'] = predicted_voice_emotion
    st.session_state['symptoms'] = symptoms
    st.session_state['advice'] = advice
    # Safe download block using session state
if (
    'face_emotion' in st.session_state and
    'voice_emotion' in st.session_state and
    'symptoms' in st.session_state and
    'advice' in st.session_state
):
    
        pdf_file = generate_pdf(
            st.session_state['face_emotion'],
            st.session_state['voice_emotion'],
            st.session_state['symptoms'],
            st.session_state['advice']
        )
        st.download_button(
        label="📄 Download Health Report as PDF",
        data=pdf_file,
        file_name="health_report.pdf",
        mime="application/pdf"


        )




   
