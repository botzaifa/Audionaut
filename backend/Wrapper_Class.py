import torch
import torchaudio
import os
import gc
import sys
import time
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade

class StemSeparator:
    def __init__(self, device=None, segment=10.0, overlap=0.1):
        self.bundle = HDEMUCS_HIGH_MUSDB_PLUS
        self.model = self.bundle.get_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else torch.device(device)
        self.model.to(self.device)
        self.sample_rate = self.bundle.sample_rate
        self.segment = segment
        self.overlap = overlap
        torch.backends.cudnn.benchmark = True  # Optimize CUDA performance
    
    def _print_progress(self, message):
        sys.stdout.write(f"\r{message}... ")
        sys.stdout.flush()

    def separate_sources(self, mix):
        batch, channels, length = mix.shape
        chunk_len = int(self.sample_rate * self.segment * (1 + self.overlap))
        start, end = 0, chunk_len
        overlap_frames = int(self.overlap * self.sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")
        final = torch.zeros(batch, len(self.model.sources), channels, length, device=self.device)
        
        # total_chunks = (length // chunk_len) + 1
        total_chunks = -(-length // chunk_len)  # Equivalent to math.ceil(length / chunk_len)
        chunk_count = 0

        while start < length - overlap_frames:
            self._print_progress(f"Processing chunk {chunk_count + 1}/{total_chunks}")
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = self.model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            start = start + chunk_len - overlap_frames if start == 0 else start + chunk_len
            end += chunk_len
            chunk_count += 1
            if end >= length:
                fade.fade_out_len = 0
        
        print("\nSeparation complete!")
        return final
    
    def process_audio(self, input_file):
        print("Loading audio file...")
        waveform, sr = torchaudio.load(input_file)
        waveform = waveform.to(self.device)
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()
        
        print("Running stem separation...")
        sources = self.separate_sources(waveform[None])
        sources = sources * ref.std() + ref.mean()
        
        return dict(zip(self.model.sources, sources[0]))
    
    def save_stems(self, audio_dict, output_dir="../data/output/"):
        os.makedirs(output_dir, exist_ok=True)
        for source_name, source_wave in audio_dict.items():
            print(f"Saving {source_name}...")
            output_path = os.path.join(output_dir, f"{source_name}.wav")
            torchaudio.save(output_path, source_wave.cpu(), self.sample_rate)
        print("All stems saved successfully!")
        
        # Free up memory after saving
        del audio_dict
        gc.collect()
        torch.cuda.empty_cache()

# Example usage
separator = StemSeparator()
stems = separator.process_audio("../data/input/song_test.wav")
separator.save_stems(stems)
