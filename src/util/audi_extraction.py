import os
from os.path import join
import moviepy.editor as mp
from tqdm import tqdm
import config as config

def extract_audio_for_videos_in_folder(video_folder_path, audio_folder_path):
    # Get all .mp4 files in the folder
    video_files = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')]

    # Extract audio for each video file
    for video_file in tqdm(video_files):
        # Get the video file path
        video_file_path = join(video_folder_path, video_file)

        # Construct the audio file path
        audio_file_name = video_file.replace('.mp4', '.wav')
        audio_file_path = join(audio_folder_path, audio_file_name)

        # Skip if the audio file already exists
        if os.path.exists(audio_file_path):
            continue

        # Extract audio
        video_clip = mp.VideoFileClip(video_file_path)
        audio_clip = video_clip.audio

        # Write audio to file
        audio_clip.write_audiofile(audio_file_path)

        # Close the video clip
        video_clip.close()

if __name__ == '__main__':
    # Extract audio for videos in the test data folder
    video_folder_path = join(config.TEST_DATA_FOLDER_PATH, 'video')
    audio_folder_path = join(config.TEST_DATA_FOLDER_PATH, 'audio')
    extract_audio_for_videos_in_folder(video_folder_path, audio_folder_path)

    # Extract audio for videos in the dev data folder
    video_folder_path = join(config.DEV_DATA_FOLDER_PATH, 'video')
    audio_folder_path = join(config.DEV_DATA_FOLDER_PATH, 'audio')
    extract_audio_for_videos_in_folder(video_folder_path, audio_folder_path)

    # Extract audio for videos in the train data folder
    video_folder_path = join(config.TRAIN_DATA_FOLDER_PATH, 'video')
    audio_folder_path = join(config.TRAIN_DATA_FOLDER_PATH, 'audio')
    extract_audio_for_videos_in_folder(video_folder_path, audio_folder_path)
