import sounddevice as sd
import numpy as np
import wave
import os
import queue
import tempfile
import threading

class AudioRecorder:
    def __init__(self, sample_rate=44100, channels=2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.q = queue.Queue()

    def callback(self, indata, frames, time, status):
        """Callback function to store recorded audio in a queue."""
        if status:
            print(status)
        self.q.put(indata.copy())

    def record_audio(self, max_duration=20):
        """Records audio for up to `max_duration` seconds, but allows stopping early."""
        temp_dir = tempfile.gettempdir()
        recorded_file = os.path.join(temp_dir, "recorded_audio.wav")

        self.recording = True
        with wave.open(recorded_file, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(self.sample_rate)

            def write_audio():
                """Write audio data to file while recording is active."""
                while self.recording:
                    try:
                        data = self.q.get(timeout=0.1)
                        wf.writeframes(data.tobytes())
                    except queue.Empty:
                        continue
            
            print(f"Recording... (Max {max_duration} sec, click 'Stop' to stop early)")
            thread = threading.Thread(target=write_audio)
            thread.start()

            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=self.callback):
                sd.sleep(max_duration * 1000)  # Convert to milliseconds

            self.recording = False
            thread.join()  # Wait for writing thread to finish

        print(f"Recording saved: {recorded_file}")
        return recorded_file

    def stop_recording(self):
        """Stops the recording immediately."""
        self.recording = False
