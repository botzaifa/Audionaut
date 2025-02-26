import librosa
import noisereduce as nr
import soundfile as sf
import sys
import os

def enhance_audio(file_path):
    """Perform noise reduction and save the enhanced audio."""
    y, sr = librosa.load(file_path, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    
    output_dir = "enhanced_audio"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "enhanced.wav")
    sf.write(output_file, reduced_noise, sr)
    print(f"Enhanced audio saved at: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python audio_enhancement.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    enhance_audio(audio_file)
