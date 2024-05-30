import cv2
import os
import numpy as np
import sys
import time
from IPython.display import display, Image, clear_output
from PIL import Image as PILImage
import io
from multiprocessing import Process, Queue, Lock
from concurrent.futures import ProcessPoolExecutor
import threading
import queue



def preprocess_frame(frame):
    # Resize the frame to 224x224
    resized_frame = cv2.resize(frame, (224, 224))
    # Convert the frame to RGB (if it's in BGR)
    if resized_frame.shape[2] == 3:
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    # Convert the frame to a format suitable for MobileNetV2
    processed_frame = resized_frame / 255.0
    # Add batch dimension
    processed_frame = np.expand_dims(processed_frame, axis=0)
    return processed_frame




def frame_reader(video_path, frame_queue, interval, fps, total_frames):
    video_capture = cv2.VideoCapture(video_path)
    frame_interval = int(fps * interval)
    
    for frame_count in range(0, total_frames, frame_interval):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_queue.put((frame_count, frame))
    
    video_capture.release()
    frame_queue.put(None)  # Signal that reading is done

def frame_processor(frame_queue, processed_queue):
    while True:
        item = frame_queue.get()
        if item is None:
            processed_queue.put(None)
            break
        frame_count, frame = item
        processed_frame = preprocess_frame(frame)
        processed_queue.put((frame_count, processed_frame))

def classify_frames(processed_queue,interval, model, sensibility, fps, porn_detected_timestamps, progress_queue, total_frames):
    porn_frame_count = 0
    frame_interval = int(fps * interval)

    while True:
        item = processed_queue.get()
        if item is None:
            break
        frame_count, processed_frame = item
        is_porn = model.predict(processed_frame, verbose=0)
        
        if is_porn >= sensibility:
            timestamp = (round(frame_count / fps,2)   ,   round((frame_count+frame_interval) / fps,2))
            porn_detected_timestamps.append(timestamp)
            porn_frame_count += 1
        
        progress_percentage = (frame_count / total_frames) * 100
        progress_queue.put(progress_percentage)
    
    progress_queue.put(None)
    return porn_frame_count

def predict_image_video(video_path, model, interval=1, void_threshold=1, sensibility=0.5700000000000001, threshold=0.1, x=''):
    start_time = time.time()
    
    # Open the video file to get metadata
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.release()

    
    frame_queue = queue.Queue(maxsize=10)
    processed_queue = queue.Queue(maxsize=10)
    progress_queue = queue.Queue()
    porn_detected_timestamps = []
    
    reader_thread = threading.Thread(target=frame_reader, args=(video_path, frame_queue, interval, fps, total_frames))
    processor_thread = threading.Thread(target=frame_processor, args=(frame_queue, processed_queue))
    classifier_thread = threading.Thread(target=classify_frames, args=(processed_queue,interval, model, sensibility, fps, porn_detected_timestamps, progress_queue, total_frames))
    
    reader_thread.start()
    processor_thread.start()
    classifier_thread.start()
    
    reader_thread.join()
    processor_thread.join()
    classifier_thread.join()
    
    

    # Calculate the ratio of pornographic frames to the total duration of the video
    porn_frame_ratio = len(porn_detected_timestamps) / (total_frames / (fps * interval)) * 100
    
    # Check if the ratio is below the threshold or if only two frames are detected as pornographic
    if porn_frame_ratio < threshold or len(porn_detected_timestamps) <= 2:
        elapsed_time = time.time() - start_time
        #clear_output(wait=True)
        print(f"    Total image time: {elapsed_time:.2f} seconds.")
        print("    Percentage of pornographic frames: 0.00%")
        return []

    smoothed_timestamps = []
    current_interval_start = None
        
    for start, end in porn_detected_timestamps:
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
    #clear_output(wait=True)
    print(f"    Total image time: {elapsed_time:.2f} seconds.")
    if porn_frame_ratio > 100 :
        porn_frame_ratio=100
    print(f"    Percentage of pornographic frames: {porn_frame_ratio:.2f}%")
    
    return smoothed_timestamps

def predict_images_in_videos_for_tests(normal_directory, porn_directory, model,seed=69, interval=1, void_threshold=1, sensibility=0.9, threshold=0.15, percentage=0.10):
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

    total_videos = len(all_videos)
    progress_message = f"Processed 0/{total_videos} videos (0.00%)"
    for idx, (video_file, label, video_path) in enumerate(all_videos):
        timestamps = classify_video_images(video_path, model, interval, void_threshold, sensibility, threshold, progress_message)
        if timestamps:
            results.append(1)
        else:
            results.append(0)

        true_labels.append(label)

        # Print progress
        progress_percentage = ((idx + 1) / total_videos) * 100
        clear_output(wait=True)
        progress_message = f"Processed {idx + 1}/{total_videos} videos ({progress_percentage:.2f}%)"
        print(progress_message)
        # Check for mismatch
        if results[-1] != true_labels[-1]:
            mismatches.append((video_file, results[-1], true_labels[-1]))

    if mismatches:
        print("Videos with mismatched results:")
        for mismatch in mismatches:
            print(f"Video: {mismatch[0]}, Predicted: {mismatch[1]}, True Label: {mismatch[2]}")
    
    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time:.2f} seconds.")

    return results, true_labels