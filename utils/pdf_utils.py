from fpdf import FPDF
import datetime
from io import BytesIO

def generate_pdf(face_emotion, voice_emotion, symptoms, advice):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title and Info
    pdf.cell(200, 10, txt="Emotion-Aware Health Report", ln=True, align="C")
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date/Time: {now}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 10, txt=f"Face Emotion: {face_emotion}", ln=True)
    pdf.cell(200, 10, txt=f"Voice Emotion: {voice_emotion}", ln=True)
    pdf.ln(10)

    # Symptoms
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="Reported Symptoms:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=symptoms)
    pdf.ln(5)

    # Advice
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt="GPT-Generated Health Advice:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=advice)

    # ✅ Convert PDF to binary string and return as BytesIO
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    pdf_stream = BytesIO(pdf_bytes)
    return pdf_stream
