from __future__ import division, print_function, absolute_import

import sys
import numpy as np
from numpy.random import seed, randint
from scipy.io import wavfile
from scipy.io.wavfile import read, WavFileWarning
from sklearn.utils.class_weight import compute_class_weight
import warnings
from pydub import AudioSegment
from pydub.utils import mediainfo
from moviepy.editor import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import time
import tempfile
import os
from preprocess_sound import preprocess_sound



# Define your preprocessing function
def preprocess_audio(audio_path):
    file_ext = os.path.splitext(audio_path)[1].lower()
    
    if file_ext == '.wav':
        sr, wav_data = wavfile.read(audio_path)
    elif file_ext == '.mp3':
        audio = AudioSegment.from_mp3(audio_path)
        sr = audio.frame_rate
        wav_data = np.array(audio.get_array_of_samples())
    else:
        raise ValueError("Unsupported audio format: Only WAV and MP3 are supported")
    cur_wav = wav_data / 32768.0
    cur_spectro = preprocess_sound(cur_wav, sr)
    cur_spectro = np.expand_dims(cur_spectro, 3)
    return cur_spectro

def predict_audio(audio_path,model):
    return model.predict(preprocess_audio(audio_path),verbose=0)

def extract_audio_from_video(video_path):
    # Load the video file
    video = VideoFileClip(video_path)
    
    # Check if the video has audio
    if video.audio is None:
        return None
    
    # Define the duration of each segment in seconds (5 minutes = 300 seconds)
    segment_duration = 300
    
    # Get the total duration of the video in seconds
    video_duration = video.duration
    
    # List to hold paths of the audio segments
    audio_paths = []
    
    # Iterate over the video duration, segmenting every 5 minutes
    for start_time in range(0, int(video_duration), segment_duration):
        # Calculate the end time of the current segment
        end_time = min(start_time + segment_duration, video_duration)
        
        # Extract the audio segment
        audio_segment = video.subclip(start_time, end_time).audio
        
        # Create a temporary file for the audio segment
        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio_path = temp_audio_file.name
        temp_audio_file.close()
        
        # Write the audio segment to the temporary file
        audio_segment.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
        
        # Append the path to the list
        audio_paths.append(temp_audio_path)
    
    return audio_paths

def predict_audio_of_video(video_path,model, frame_duration=0.96, sensibility=0.9, threshold=0.1, void_threshold=0.96, audio_threshold=-50):
    start_time = time.time()
    audio_paths = extract_audio_from_video(video_path)
    
    if audio_paths is None:
        elapsed_time = time.time() - start_time
        print(f"    Total audio time: {elapsed_time:.2f} seconds.")
        print("    No Audio")
        return None
    
    porn_detected_indices = []
    total_frame_count = 0
    
    try:
        for segment_index, audio_temp_path in enumerate(audio_paths):
            audio = AudioSegment.from_file(audio_temp_path)
            
            # Calculate the average dBFS (Decibels relative to Full Scale)
            audio_dBFS = audio.dBFS
            
            if audio_dBFS < audio_threshold:
                print('    Audio too quiet')
                continue
            
            predictions = predict_audio(audio_temp_path,model)
            segment_porn_frame_count = 0
            for index, prediction in enumerate(predictions):
                if prediction > sensibility:
                    segment_porn_frame_count += 1
                    start = round(index * frame_duration + segment_index * 300, 2)
                    end = round(start + frame_duration, 2)
                    porn_detected_indices.append((start, end))
            
            total_frame_count += len(predictions)
            os.remove(audio_temp_path)
        
        # Calculate the ratio of pornographic frames to the total number of frames
        porn_frame_ratio = (len(porn_detected_indices) / total_frame_count) * 100 if total_frame_count > 0 else 0
        
        # Check if the ratio is below the threshold or if only two frames are detected as pornographic
        if porn_frame_ratio < (threshold * 100) or len(porn_detected_indices) <= 2:
            elapsed_time = time.time() - start_time
            print(f"    Total audio time: {elapsed_time:.2f} seconds.")
            print("    Percentage of pornographic audios: 0.00%")
            return []
        
        smoothed_timestamps = []
        current_interval_start = None
        
        for start, end in porn_detected_indices:
            if current_interval_start is None:
                current_interval_start = start
                current_interval_end = end
            elif start - current_interval_end <= void_threshold:
                # Merge with the current interval
                current_interval_end = max(current_interval_end, end)
            else:
                # Add the current interval to smoothed_timestamps
                smoothed_timestamps.append((current_interval_start, current_interval_end))
                # Start a new interval
                current_interval_start = start
                current_interval_end = end
        
        # Add the last interval if it exists
        if current_interval_start is not None:
            smoothed_timestamps.append((current_interval_start, current_interval_end))
        
        elapsed_time = time.time() - start_time
        print(f"    Total audio time: {elapsed_time:.2f} seconds.")
        if porn_frame_ratio > 100 :
            porn_frame_ratio = 100
        print(f"    Percentage of pornographic audio: {porn_frame_ratio:.2f}%")
        return smoothed_timestamps
    
    except Exception as e:
        # Delete any remaining temporary files
        for path in audio_paths:
            if os.path.exists(path):
                os.remove(path)
        elapsed_time = time.time() - start_time
        print(f"    Total audio time: {elapsed_time:.2f} seconds.")
        print('    Exception:', e)
        return None

def predict_audio_in_videos_for_tests(normal_directory, porn_directory,model,seed=69, void_threshold=2, sensibility=0.98, threshold=0.1, percentage=0.10):
    start_time = time.time()
    np.random.seed(seed)  # For reproducibility

    normal_videos = [filename for filename in os.listdir(normal_directory) if filename.endswith(".mp4")]
    porn_videos = [filename for filename in os.listdir(porn_directory) if filename.endswith(".mp4")]

    # Determine the number of videos to test based on the given percentage
    num_normal_videos_to_test = int(len(normal_videos) * percentage)
    num_porn_videos_to_test = int(len(porn_videos) * percentage)

    # Randomly select videos to test
    normal_videos_to_test = np.random.choice(normal_videos, num_normal_videos_to_test, replace=False)
    porn_videos_to_test = np.random.choice(porn_videos, num_porn_videos_to_test, replace=False)

    all_videos = [(video, 0, os.path.join(normal_directory, video)) for video in normal_videos_to_test] + \
                 [(video, 1, os.path.join(porn_directory, video)) for video in porn_videos_to_test]

    results = []
    true_labels = []
    mismatches = []
    too_quiets = []

    total_videos = len(all_videos)
    progress_message = f"Processed 0/{total_videos} videos (0.00%)"
    for idx, (video_file, label, video_path) in enumerate(all_videos):
        timestamps = predict_audio_of_video(video_path,model ,sensibility=sensibility, threshold=threshold, void_threshold=void_threshold)
        if timestamps is not None :
            if len(timestamps)>0:
                results.append(1)
            else:
                results.append(0)
        else :
            results.append(None)
            too_quiets.append(video_file)

        true_labels.append(label)

        # Print progress
        progress_percentage = ((idx + 1) / total_videos) * 100
        clear_output(wait=True)
        progress_message = f"Processed {idx + 1}/{total_videos} videos ({progress_percentage:.2f}%)"
        print(progress_message)
        # Check for mismatch
        if results[-1] != true_labels[-1]:
            if results[-1] is not None :
                mismatches.append((video_file, results[-1], true_labels[-1]))

    if mismatches:
        print("Videos with mismatched results :")
        for mismatch in mismatches:
            print(f"Video: {mismatch[0]}, Predicted: {mismatch[1]}, True Label: {mismatch[2]}")
    if too_quiets:
        print("Videos with no audio, couldn't read the audio successfully or too quiet :")
        for too_quiet in too_quiets:
            print(f"Video: {too_quiet}.")
    
    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time:.2f} seconds.")

    return results, true_labels


def clear_none_from_predictions(predictions, trueLabels):
    indexes = []
    for index, item in enumerate(predictions):
        if item is None:
            indexes.append(index)
    if indexes:
        for index in sorted(indexes, reverse=True):
            predictions.pop(index)
            trueLabels.pop(index)
    return predictions, trueLabels