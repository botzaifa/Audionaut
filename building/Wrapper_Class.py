import torch
import torchaudio
import matplotlib.pyplot as plt
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
import os

class StemSeparator:
    def __init__(self, device=None):
        self.bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.model = self.bundle.get_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.model.to(self.device)
        self.sample_rate = self.bundle.sample_rate
        
        # Set STFT transformation parameters
        self.N_FFT = 4096
        self.N_HOP = 512
        self.stft_transform = torchaudio.transforms.Spectrogram(n_fft=self.N_FFT, hop_length=self.N_HOP, power=None).to(self.device)
    
    def separate_sources(self, mix, segment=10.0, overlap=0.1):
        batch, channels, length = mix.shape
        chunk_len = int(self.sample_rate * segment * (1 + overlap))
        start, end = 0, chunk_len
        overlap_frames = int(overlap * self.sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")
        final = torch.zeros(batch, len(self.model.sources), channels, length, device=self.device)
        
        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = self.model.forward(chunk)
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
    
    def plot_spectrogram(self, waveform, title="Spectrogram"):
        waveform = waveform.mean(dim=0, keepdim=True).to(self.device)  # Convert to mono for visualization
        magnitude = self.stft_transform(waveform).abs()
        spectrogram_db = 20 * torch.log10(magnitude + 1e-8).squeeze().cpu().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(spectrogram_db, cmap="magma", aspect="auto", origin="lower")
        plt.title(title)
        plt.xlabel("Time Frames")
        plt.ylabel("Frequency Bins")
        plt.colorbar(label="Amplitude (dB)")
        plt.tight_layout()
        plt.show()
    
    def process_audio(self, input_file):
        waveform, sr = torchaudio.load(input_file)
        waveform = waveform.to(self.device)
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()
        
        sources = self.separate_sources(waveform[None])
        sources = sources * ref.std() + ref.mean()
        
        sources_list = self.model.sources
        sources = list(sources[0])
        self.audios = dict(zip(sources_list, sources))
        
        for source_name, source_wave in self.audios.items():
            print(f"Displaying {source_name} stem")
            self.plot_spectrogram(source_wave, f"Spectrogram - {source_name}")
        
        return self.audios
    
    def save_stems(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for source_name, source_wave in self.audios.items():
            output_path = os.path.join(output_dir, f"{source_name}.wav")
            torchaudio.save(output_path, source_wave.cpu(), self.sample_rate)
            print(f"Saved {source_name} to {output_path}")

# Example usage
# separator = StemSeparator()
# stems = separator.process_audio("path/to/file.wav")
# separator.save_stems("output_directory")
