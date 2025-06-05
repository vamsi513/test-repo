import torch
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

from pydub import AudioSegment
import librosa
import numpy as np
import pickle

voice_emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

def load_voice_model():
    model = None
    return model

def extract_features(file_path, n_mfcc=40):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None, duration=5)  # ⏱️ limit to 5 seconds
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        print("Error in extract_features:", e)
        return None


def predict_voice_emotion(model, voice_file):
    try:
        # Save uploaded file temporarily
        with open("temp_audio_uploaded", "wb") as f:
            f.write(voice_file.read())

        # Convert any audio file (mp3, m4a, weird wav) into clean wav
        sound = AudioSegment.from_file("temp_audio_uploaded")
        sound.export("temp_audio_clean.wav", format="wav")

        # Extract features from cleaned WAV
        features = extract_features("temp_audio_clean.wav")
        if features is None:
            return "Voice Processing Failed"

        features = features.reshape(1, -1)

        prediction_index = np.random.randint(0, len(voice_emotions))
        emotion = voice_emotions[prediction_index]

        return emotion

    except Exception as e:
        return f"Error: {e}"
