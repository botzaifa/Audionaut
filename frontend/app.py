import streamlit as st
import os
import subprocess
import librosa
import soundfile as sf
import numpy as np
import tempfile
from pathlib import Path

# Set up the Streamlit app
st.title("🎵 Audio Processing App")
st.sidebar.header("Select an Operation")

# Choose between Stem Separation and Audio Enhancement
option = st.sidebar.radio("Choose a function:", ["Stem Separation", "Audio Enhancement"])

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    input_audio_path = os.path.join(temp_dir.name, uploaded_file.name)
    
    with open(input_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(input_audio_path, format="audio/wav", start_time=0)

    if option == "Stem Separation":
        st.subheader("🎤 Stem Separation")
        if st.button("Process Audio"):
            with st.spinner("Processing..."):
                subprocess.run(["python", "stem_separation.py", input_audio_path])
            
            # Display separated stems
            sources = ["drums_stem.wav", "bass_stem.wav", "other_stem.wav", "vocals_stem.wav"]
            for source in sources:
                if Path(source).exists():
                    st.audio(source, format="audio/wav")
                    with open(source, "rb") as f:
                        st.download_button(f"Download {source}", f, file_name=source)

    elif option == "Audio Enhancement":
        st.subheader("🔊 Audio Enhancement")
        if st.button("Enhance Audio"):
            with st.spinner("Enhancing..."):
                subprocess.run(["python", "audio_enhancement.py", input_audio_path])

            output_file = "enhanced_audio.wav"
            if Path(output_file).exists():
                st.audio(output_file, format="audio/wav")
                with open(output_file, "rb") as f:
                    st.download_button("Download Enhanced Audio", f, file_name=output_file)

    temp_dir.cleanup()
