import streamlit as st
import os
import sys
from pathlib import Path

# Fix Import Issue: Add 'backend/' to Python Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import backend modules AFTER fixing path
from backend.stem import separate_stems
from backend.enhancement import enhance_audio
from backend.rec_aud import start_recording, pause_recording, resume_recording, stop_recording

# Define directories
DATA_INPUT_DIR = "../data/input/"
DATA_OUTPUT_DIR = "../data/output/"
os.makedirs(DATA_INPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# Initialize session state for selected file
if "selected_file" not in st.session_state:
    st.session_state["selected_file"] = None

# Streamlit App UI
st.title("🎵 Audionaut: Audio Processing App")
st.sidebar.header("Select an Operation")

# Select Operation
option = st.sidebar.radio("Choose a function:", ["Stem Separation", "Audio Enhancement"])

# File selection: Upload OR Select from existing
st.subheader("📂 Select an Audio File")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
existing_files = [f for f in os.listdir(DATA_INPUT_DIR) if f.endswith((".wav", ".mp3"))]

selected_file = st.session_state["selected_file"]

if existing_files:
    selected_option = st.selectbox("Or select an existing file:", ["None"] + existing_files)
    if selected_option != "None":
        st.session_state["selected_file"] = selected_option

# Handle uploaded file
if uploaded_file:
    temp_file_path = os.path.join(DATA_INPUT_DIR, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["selected_file"] = uploaded_file.name

# 🎤 Recording Feature (Only for Audio Enhancement)
if option == "Audio Enhancement":
    st.subheader("🎤 Record Audio Instead")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("▶️ Start Recording"):
            start_recording()
            st.success("Recording started...")

    with col2:
        if st.button("⏸️ Pause Recording"):
            pause_recording()
            st.warning("Recording paused...")

    with col3:
        if st.button("▶️ Resume Recording"):
            resume_recording()
            st.info("Recording resumed...")
            
    with col4:
        if st.button("⏹️ Stop Recording"):
            recorded_audio = stop_recording()
            if recorded_audio:
                recorded_audio_path = os.path.join(DATA_INPUT_DIR, os.path.basename(recorded_audio))
                st.session_state["selected_file"] = os.path.basename(recorded_audio)
                st.session_state["just_recorded"] = True  # ✅ Set flag to prevent duplicate playback
                st.success(f"Recording stopped and saved as {st.session_state['selected_file']}.")
                st.audio(recorded_audio_path, format="audio/wav")  # ✅ Play only here


# Proceed with processing if a file is available
if st.session_state["selected_file"]:
    file_path = os.path.join(DATA_INPUT_DIR, st.session_state["selected_file"])
    st.audio(file_path, format="audio/wav")

    if option == "Stem Separation":
        st.subheader("🎤 Stem Separation")
                        
        # Karaoke Mode Toggle
        karaoke_mode = st.checkbox("🎤 Enable Karaoke Mode (Remove Vocals)")

        if st.button("Process Audio"):
            with st.spinner("Processing..."):
                stems = separate_stems(file_path, DATA_OUTPUT_DIR, karaoke_mode)
            
            st.success("✅ Stem separation complete! Download the processed files below:")

            # If Karaoke Mode, show only the instrumental file
            if karaoke_mode:
                st.audio(stems["instrumental"], format="audio/wav")
                with open(stems["instrumental"], "rb") as f:
                    st.download_button("Download Instrumental", f, file_name="instrumental.wav")
            else:
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
    st.warning("⚠️ Please upload, select, or record a file to proceed.")
