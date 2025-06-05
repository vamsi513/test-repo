# utils/nlp_utils.py

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load the OpenAI API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client (new SDK style)
client = OpenAI(api_key=api_key)

def generate_health_advice(symptoms_text):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful healthcare assistant. Provide preliminary advice based on symptoms, but remind the user to consult a doctor for serious concerns."
                },
                {
                    "role": "user",
                    "content": f"My symptoms are: {symptoms_text}"
                }
            ],
            temperature=0.5,
            max_tokens=300,
            n=1,
        )

        advice = response.choices[0].message.content.strip()
        return advice

    except Exception as e:
        return f"Error in generating advice: {e}"
