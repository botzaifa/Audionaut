import torch
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Load HDemucs model
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
sample_rate = bundle.sample_rate
print(f"Sample rate: {sample_rate}")

def separate_sources(model, mix, segment=10.0, overlap=0.1, device=None):
    """
    Apply model to a given mixture. Use fade, and add segments together.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)
    
    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start, end = 0, chunk_len
    overlap_frames = int(overlap * sample_rate)
    fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")
    final = torch.zeros(batch, len(model.sources), channels, length, device=device)
    
    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = overlap_frames
            start += chunk_len - overlap_frames
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final

def plot_spectrogram(stft_transform, waveform, title="Spectrogram"):
    """Plots a 2D spectrogram with Time (X-axis) and Frequency (Y-axis)."""
    waveform = waveform.mean(dim=0, keepdim=True).to(device)  # Convert to mono for visualization
    magnitude = stft_transform(waveform).abs()  # Convert complex STFT to magnitude

    # Convert to decibels (log scale)
    spectrogram_db = 20 * torch.log10(magnitude + 1e-8).squeeze().cpu().numpy()

    # Plot 2D spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram_db, cmap="magma", aspect="auto", origin="lower")

    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()

    plt.show()
# Set STFT transformation parameters
N_FFT = 4096
N_HOP = 512  # Adjusted hop size for visualization
stft_transform = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=N_HOP, power=None).to(device)

# User input file path
input_file = input("Enter the path to the WAV file: ")
waveform, sr = torchaudio.load(input_file)
waveform = waveform.to(device)

# Normalize audio
ref = waveform.mean(0)
waveform = (waveform - ref.mean()) / ref.std()

# Separate sources
sources = separate_sources(model, waveform[None], device=device)[0]
sources = sources * ref.std() + ref.mean()

# Extract and display results
sources_list = model.sources
sources = list(sources)
audios = dict(zip(sources_list, sources))

for source_name, source_wave in audios.items():
    print(f"Displaying {source_name} stem")
    
    # Plot spectrogram
    plot_spectrogram(stft_transform, source_wave, f"Spectrogram - {source_name}")

    # Provide audio playback option
    display(Audio(source_wave.cpu().numpy(), rate=sample_rate))
