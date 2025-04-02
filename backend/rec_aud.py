import pyaudio
import wave
import threading
import os

# Recording Parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILENAME = "../data/input/recorded_audio.wav"  # Save directly to input folder

# Global control variables
is_recording = False
is_paused = False
frames = []
audio = pyaudio.PyAudio()

def record():
    global is_recording, is_paused, frames
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while is_recording:
        if not is_paused:
            frames.append(stream.read(CHUNK))

    stream.stop_stream()
    stream.close()

def start_recording():
    global is_recording, frames
    is_recording = True
    frames = []
    threading.Thread(target=record).start()

def pause_recording():
    global is_paused
    is_paused = True

def resume_recording():
    global is_paused
    is_paused = False

def stop_recording():
    global is_recording
    is_recording = False

    # Save the recorded file
    with wave.open(OUTPUT_FILENAME, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return OUTPUT_FILENAME  # Return path for selection
