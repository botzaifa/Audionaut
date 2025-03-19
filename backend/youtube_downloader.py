# import yt_dlp

# def download_youtube_video(video_url, output_path="."):
#     """Downloads the YouTube video as an MP4 file using yt-dlp."""
#     try:
#         ydl_opts = {
#             'outtmpl': f"{output_path}/%(title)s.%(ext)s",  # Save as video title
#             'format': 'bestvideo+bestaudio/best',  # Get the best quality
#             'merge_output_format': 'mp4',  # Ensure MP4 format
#         }

#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             ydl.download([video_url])

#         print("✅ Download Complete!")
#     except Exception as e:
#         print(f"❌ Error: {e}")

# # Get user input and download
# video_url = input("Enter the YouTube video URL: ").strip()
# download_youtube_video(video_url)
