# import streamlit as st
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import soundfile as sf
# import io

# # Function to process audio (Replace this with your actual backend function)
# def process_audio(audio_data, sample_rate):
#     # Example: Simple noise reduction (Replace with your actual processing logic)
#     processed_audio = audio_data * 0.8  # Dummy processing
#     return processed_audio

# st.title("🎵 Audio Processing App")
# st.write("Upload an audio file, process it, and listen to the enhanced version!")

# uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

# if uploaded_file is not None:
#     # Load the uploaded file
#     audio, sr = librosa.load(uploaded_file, sr=None)
    
#     st.audio(uploaded_file, format='audio/wav')
#     st.write("### Original Audio Waveform")
#     fig, ax = plt.subplots()
#     librosa.display.waveshow(audio, sr=sr, ax=ax)
#     st.pyplot(fig)
    
#     # Process the audio
#     processed_audio = process_audio(audio, sr)
    
#     st.write("### Processed Audio Waveform")
#     fig, ax = plt.subplots()
#     librosa.display.waveshow(processed_audio, sr=sr, ax=ax)
#     st.pyplot(fig)
    
#     # Save processed audio to a buffer
#     buffer = io.BytesIO()
#     sf.write(buffer, processed_audio, sr, format='WAV')
#     buffer.seek(0)
    
#     st.audio(buffer, format='audio/wav')
    
#     st.download_button("Download Processed Audio", buffer, file_name="processed_audio.wav", mime="audio/wav")

import streamlit as st
import os
import subprocess

st.title("🎵 Audio Processing App")

# Navigation
page = st.sidebar.radio("Choose a Task", ["Home", "Stem Separation", "Audio Enhancement"])

if page == "Home":
    st.write("### Welcome to the Audio Processing App! 🎧")
    st.write("Choose a task from the sidebar to get started.")

elif page == "Stem Separation":
    st.write("## 🎼 Stem Separation")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
        
        if st.button("Process Stems"):
            subprocess.run(["python", "stem_separation.py", file_path])
            st.success("Stem separation complete! Check output folder.")

elif page == "Audio Enhancement":
    st.write("## 🎛️ Audio Enhancement")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
        
        if st.button("Enhance Audio"):
            subprocess.run(["python", "audio_enhancement.py", file_path])
            st.success("Audio enhancement complete! Check output folder.")
