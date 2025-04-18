import streamlit as st
import os
import sys
from pathlib import Path

# Fix Import Issue: Add 'backend/' to Python Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import backend modules
from backend.stem import separate_stems
from backend.enhancement import enhance_audio
from backend.rec_aud import start_recording, pause_recording, resume_recording, stop_recording

# Directories
DATA_INPUT_DIR = "../data/input/"
DATA_OUTPUT_DIR = "../data/output/"
os.makedirs(DATA_INPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# Session State Initialization
if "selected_file" not in st.session_state:
    st.session_state["selected_file"] = None
if "just_recorded" not in st.session_state:
    st.session_state["just_recorded"] = False

# App Title
st.title("🎵 Audionaut: Audio Processing App")
st.sidebar.header("Select an Operation")

# Operation Selector
option = st.sidebar.radio("Choose a function:", ["Stem Separation", "Audio Enhancement"])

# Audio Input Section
st.subheader("📂 Select an Audio File")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

existing_files = [f for f in os.listdir(DATA_INPUT_DIR) if f.endswith((".wav", ".mp3"))]
selected_option = st.selectbox("Or select an existing file:", ["None"] + existing_files)

if selected_option != "None":
    st.session_state["selected_file"] = selected_option
    st.session_state["just_recorded"] = False

if uploaded_file:
    uploaded_path = os.path.join(DATA_INPUT_DIR, uploaded_file.name)
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state["selected_file"] = uploaded_file.name
    st.session_state["just_recorded"] = False

# 🎤 Recording Section
recorded_audio_path = None
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
                st.session_state["just_recorded"] = True
                st.success(f"Recording saved as {st.session_state['selected_file']}.")

# Load and display the audio preview once
final_input_file = st.session_state["selected_file"]
if final_input_file:
    final_input_path = os.path.join(DATA_INPUT_DIR, final_input_file)

    if option == "Audio Enhancement" and st.session_state["just_recorded"]:
        st.audio(final_input_path, format="audio/wav")
        st.session_state["just_recorded"] = False  # Reset so it doesn’t play again

    elif option == "Stem Separation" or not st.session_state["just_recorded"]:
        st.audio(final_input_path, format="audio/wav")

    # 👇 Task-specific Actions
    if option == "Stem Separation":
        st.subheader("🎤 Stem Separation")

        karaoke_mode = st.checkbox("🎤 Enable Karaoke Mode (Remove Vocals)")

        if st.button("Process Audio"):
            with st.spinner("Processing..."):
                stems = separate_stems(final_input_path, DATA_OUTPUT_DIR, karaoke_mode)

            st.success("✅ Stem separation complete! Download the processed files below:")

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

        # Get recorded file path if exists and was just recorded
        is_recorded = st.session_state.get("just_recorded", False)
        recorded_path = os.path.join(DATA_INPUT_DIR, st.session_state["selected_file"]) if is_recorded else None
        enhancement_input_path = recorded_path if is_recorded else final_input_path

        if st.button("Enhance Audio"):
            with st.spinner("Enhancing..."):
                output_file = os.path.join(DATA_OUTPUT_DIR, "enhanced_audio.wav")
                enhanced_path = enhance_audio(enhancement_input_path, output_file)

            st.success("✅ Audio enhancement complete! Download the enhanced file below:")
            st.audio(enhanced_path, format="audio/wav")
            with open(enhanced_path, "rb") as f:
                st.download_button("Download Enhanced Audio", f, file_name="enhanced_audio.wav")


else:
    st.warning("⚠️ Please upload, select, or record a file to proceed.")
