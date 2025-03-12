from df.enhance import enhance, init_df, load_audio, save_audio
import warnings
import sys
import os

warnings.filterwarnings("ignore")

class AudioDenoiser:
    def __init__(self):
        """
        Initialize the DeepFilterNet model for audio enhancement.
        """
        self.model, self.df_state, _ = init_df()
    
    def denoise_audio(self, audio_path: str, output_path: str):
        """
        Denoises the given audio file and saves the output.
        
        :param audio_path: Path to the input noisy audio file.
        :param output_path: Path to save the enhanced audio file.
        """
        print(f"Loading audio: {audio_path}")
        audio, _ = load_audio(audio_path, sr=self.df_state.sr())
        print("Enhancing audio...")
        enhanced_audio = enhance(self.model, self.df_state, audio)
        save_audio(output_path, enhanced_audio, self.df_state.sr())
        print(f"Enhanced audio saved at: {output_path}")
        return output_path  # Return path for Streamlit

# Function for Streamlit
def enhance_audio(input_audio, output_audio):
    denoiser = AudioDenoiser()
    return denoiser.denoise_audio(input_audio, output_audio)

if __name__ == "__main__":
    input_audio = sys.argv[1]
    output_audio = sys.argv[2]
    enhance_audio(input_audio, output_audio)
