import streamlit as st
import os
import sys
from pathlib import Path

# Fix Import Issue: Add 'backend/' to Python Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import backend modules AFTER fixing path
from backend.stem import separate_stems
from backend.enhancement import enhance_audio

# Define directories
DATA_INPUT_DIR = "../data/input/"
DATA_OUTPUT_DIR = "../data/output/"
os.makedirs(DATA_INPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# Streamlit App UI
st.title("🎵 Audionaut: Audio Processing App")
st.sidebar.header("Select an Operation")

# Select Operation
option = st.sidebar.radio("Choose a function:", ["Stem Separation", "Audio Enhancement"])

# File selection: Upload OR Select from existing
st.subheader("📂 Select an Audio File")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
existing_files = [f for f in os.listdir(DATA_INPUT_DIR) if f.endswith((".wav", ".mp3"))]

selected_file = None
if existing_files:
    selected_file = st.selectbox("Or select an existing file:", ["None"] + existing_files)

# Get the final file path
if uploaded_file:
    temp_file_path = os.path.join(DATA_INPUT_DIR, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    selected_file = uploaded_file.name
elif selected_file == "None":
    selected_file = None

if selected_file:
    file_path = os.path.join(DATA_INPUT_DIR, selected_file)
    st.audio(file_path, format="audio/wav")

    if option == "Stem Separation":
        st.subheader("🎤 Stem Separation")
        if st.button("Process Audio"):
            with st.spinner("Processing..."):
                stems = separate_stems(file_path, DATA_OUTPUT_DIR)
            
            st.success("✅ Stem separation complete! Download the separated files below:")
            for stem_name, stem_path in stems.items():
                st.audio(stem_path, format="audio/wav")
                with open(stem_path, "rb") as f:
                    st.download_button(f"Download {stem_name}.wav", f, file_name=f"{stem_name}.wav")

    elif option == "Audio Enhancement":
        st.subheader("🔊 Audio Enhancement")
        if st.button("Enhance Audio"):
            with st.spinner("Enhancing..."):
                output_file = os.path.join(DATA_OUTPUT_DIR, "enhanced_audio.wav")
                enhanced_path = enhance_audio(file_path, output_file)

            st.success("✅ Audio enhancement complete! Download the enhanced file below:")
            st.audio(enhanced_path, format="audio/wav")
            with open(enhanced_path, "rb") as f:
                st.download_button("Download Enhanced Audio", f, file_name="enhanced_audio.wav")

else:
    st.warning("⚠️ Please upload or select a file to proceed.")
