# import os
# import subprocess

# class AudioConverter:
#     def __init__(self):
#         self.ffmpeg_cmd = "ffmpeg"  # Ensure ffmpeg is installed

#     def convert_to_mp3(self, input_path: str, output_dir="../data/input/") -> str:
#         """
#         Converts any audio/video file to MP3 format.

#         :param input_path: Path to the input file (audio/video).
#         :param output_dir: Directory to save the converted file.
#         :return: Path to the converted MP3 file.
#         """
#         os.makedirs(output_dir, exist_ok=True)
#         output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + ".mp3")

#         command = [self.ffmpeg_cmd, "-i", input_path, "-vn", "-acodec", "libmp3lame", output_path]
#         subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#         return output_path
