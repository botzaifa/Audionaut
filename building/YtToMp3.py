import os
from pytube import YouTube
from moviepy.editor import *  # Import moviepy library

def download_and_convert_to_mp3(url, output_path='downloads'):
    # Create a directory to save the video and MP3 files
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Download the YouTube video
    try:
        print("Downloading video...")
        yt = YouTube(url)
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()  # Choose the best video stream
        video_path = video_stream.download(output_path=output_path)
        print(f"Video downloaded successfully: {video_path}")
    except Exception as e:
        print(f"Error downloading video: {e}")
        return

    # Convert video to MP3
    try:
        print("Converting video to MP3...")
        video_clip = AudioFileClip(video_path)  # Extract audio from the video
        mp3_path = os.path.join(output_path, yt.title + '.mp3')  # Save the MP3 file with the video title as name
        video_clip.write_audiofile(mp3_path)
        video_clip.close()
        print(f"Conversion to MP3 successful: {mp3_path}")

        # Optionally, remove the original video after conversion
        os.remove(video_path)
        print(f"Original video removed: {video_path}")
    except Exception as e:
        print(f"Error during conversion: {e}")

# Example usage
url = input("Enter the YouTube video URL: ")
download_and_convert_to_mp3(url)
