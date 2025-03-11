from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

class AudioDenoiser:
    def __init__(self):
        """
        Initialize the DeepFilterNet model for audio enhancement.
        """
        self.model, self.df_state, _ = init_df()
    
    def denoise_audio(self, audio_path: str, output_path: str = "enhanced.wav"):
        """
        Denoises the given audio file and saves the output.
        
        :param audio_path: Path to the input noisy audio file.
        :param output_path: Path to save the enhanced audio file.
        """
        audio, _ = load_audio(audio_path, sr=self.df_state.sr())
        enhanced_audio = enhance(self.model, self.df_state, audio)
        save_audio(output_path, enhanced_audio, self.df_state.sr())
        print(f"Enhanced audio saved at: {output_path}")

# Example usage
if __name__ == "__main__":
    denoiser = AudioDenoiser()
    input_audio_path = "Noise Test.wav"  # Replace with your file path
    denoiser.denoise_audio(input_audio_path)
