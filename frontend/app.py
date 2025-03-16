import streamlit as st
import os
import sys
import subprocess
import tempfile
from pathlib import Path

# Fix import issues by adding the project root to sys.path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.converter import convert_to_mp3  # MP3 Converter
from backend.enhancement import AudioDenoiser  # Audio Enhancement
from backend.stem import StemSeparator  # Stem Separation
from backend.recorder import AudioRecorder  # Recorder

# Set up Streamlit app
st.title("🎵 Audionaut: Audio Processing App")
st.sidebar.header("Select an Operation")

# Choose between Stem Separation and Audio Enhancement
option = st.sidebar.radio("Choose a function:", ["Stem Separation", "Audio Enhancement"])

# Upload audio/video file
uploaded_file = st.file_uploader("Upload an audio/video file", type=["wav", "mp3", "mp4", "m4a", "flac", "aac"])

# Create temporary directory
temp_dir = tempfile.TemporaryDirectory()
data_output_dir = os.path.join(os.getcwd(), "data", "output")
os.makedirs(data_output_dir, exist_ok=True)

if uploaded_file:
    # Save uploaded file
    input_audio_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(input_audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert to MP3 if needed
    converted_audio_path = convert_to_mp3(input_audio_path)

    st.success(f"File processed: {Path(converted_audio_path).name}")
    st.audio(converted_audio_path, format="audio/mp3")

    if option == "Stem Separation":
        st.subheader("🎤 Stem Separation")
        if st.button("Process Audio"):
            with st.spinner("Processing..."):
                separator = StemSeparator()
                stems = separator.process_audio(converted_audio_path)
                separator.save_stems(stems, data_output_dir)

            st.success("Stem separation complete! Download the separated files below:")
            for stem in stems.keys():
                stem_path = os.path.join(data_output_dir, f"{stem}.wav")
                with open(stem_path, "rb") as f:
                    st.download_button(f"Download {stem}.wav", f, file_name=f"{stem}.wav", mime="audio/wav")

    elif option == "Audio Enhancement":
        st.subheader("🔊 Audio Enhancement")

        # Recorder Section (Only for Audio Enhancement)
        st.write("🎤 **Record Audio (Max 20 sec, Stop Anytime)**")
        if st.button("Start Recording"):
            recorder = AudioRecorder()
            recorded_audio_path = recorder.record_audio()
            st.success("Recording complete!")
            st.audio(recorded_audio_path, format="audio/wav")

            # Use recorded file instead of uploaded file
            converted_audio_path = recorded_audio_path

        if st.button("Enhance Audio"):
            with st.spinner("Enhancing..."):
                output_file = os.path.join(data_output_dir, "enhanced_audio.wav")
                denoiser = AudioDenoiser()
                denoiser.denoise_audio(converted_audio_path, output_file)

            st.success("Audio enhancement complete! Download below:")
            with open(output_file, "rb") as f:
                st.download_button("Download Enhanced Audio", f, file_name="enhanced_audio.wav", mime="audio/wav")

# Cleanup temporary directory
temp_dir.cleanup()
