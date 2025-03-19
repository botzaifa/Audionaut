# import os
# import tempfile
# from pytube import YouTube
# from pydub import AudioSegment

# class YouTubeDownloader:
#     def __init__(self, output_dir="../data/input/"):
#         self.output_dir = output_dir
#         os.makedirs(self.output_dir, exist_ok=True)

#     def download_and_convert(self, url):
#         """Downloads a YouTube video and converts it to an MP3 file."""
#         try:
#             # Step 1: Download YouTube video
#             yt = YouTube(url)
#             stream = yt.streams.filter(only_audio=True).first()  # Get best audio stream

#             if not stream:
#                 raise Exception("No audio streams found!")

#             # Step 2: Save to a temporary file
#             temp_dir = tempfile.mkdtemp()
#             temp_file_path = os.path.join(temp_dir, "temp_audio.mp4")
#             stream.download(output_path=temp_dir, filename="temp_audio.mp4")

#             # Step 3: Convert to MP3
#             audio = AudioSegment.from_file(temp_file_path, format="mp4")
#             mp3_file_path = os.path.join(self.output_dir, f"{yt.title}.mp3")
#             audio.export(mp3_file_path, format="mp3")

#             # Clean up temporary files
#             os.remove(temp_file_path)

#             return mp3_file_path  # Return path of the converted file

#         except Exception as e:
#             return f"Error: {str(e)}"
