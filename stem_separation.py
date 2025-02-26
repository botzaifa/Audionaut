import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import soundfile as sf
import sys

# Load the Demucs model for stem separation
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
sample_rate = bundle.sample_rate

def separate_stems(audio_path):
    """Performs stem separation on an input audio file."""
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=sample_rate)
    waveform = waveform.to(device)

    # Perform stem separation
    with torch.no_grad():
        stems = model(waveform[None])[0]

    sources = ["drums", "bass", "other", "vocals"]
    for i, source in enumerate(sources):
        output_path = f"{source}_stem.wav"
        sf.write(output_path, stems[i].cpu().numpy().T, sample_rate)
        print(f"Saved {source} stem as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stem_separation.py <audio_file>")
    else:
        separate_stems(sys.argv[1])
