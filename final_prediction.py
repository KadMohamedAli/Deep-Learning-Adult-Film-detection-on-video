from porn_audios_detector import predict_audio_of_video,clear_none_from_predictions
from porn_images_detector import predict_image_video
# make sure porn_audios_detector and porn_images_detector are correctly imported

from IPython.display import clear_output
import time
import numpy as np
import os




def predict(video_path , image_model , audio_model , option = 3 , verbose = True):
    # args :
    #     video_path : path of the video to predict. tested on : .mp4 .avi (not tested with others, could work).
    #     image_model : image model.
    #     audio_model : sound model.
    #     option : option for how to combine results 
    #         if 3 : return the two results seperatly image/audio (default choice).
    #         if 4 : return two combined results union/intersection.
    #         otherwise, return a single result.
    #         Imges results are more pertinent, may depend on the video content tho 
    #     verbose : True by default, if False, won't print anything
    #
    # return :
    #     Array of pair of tuple where the first element in the tuple represents the start of the detected pornography
    #   and the second element represents the end of the detected pornography
    #   in seconds.
    #     If return is None, there is a problem somewhere, should print it exactly.

    def print_if_verbose(message):
        if verbose:
            print(message)
    
    #check if option is correct
    if option not in (1, 2, 3, 4):
        print_if_verbose('Invalid option...')
        print_if_verbose('Available options are :')
        print_if_verbose('    1 -> union of results.')
        print_if_verbose('    2 -> intersection of results.')
        print_if_verbose('    3 -> seperate results.')
        print_if_verbose('    3 -> union + intersection of results.')
        return None

    #supported extensions 
    #only tested with mp4 and avi but should works with others extensions
    #video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    video_extensions = {'.mp4', '.avi'}
    
    is_video_legit , message =is_video_file(video_path , video_extensions)

    if not is_video_legit :
        print_if_verbose(message)
        return None

    file_name, _ = os.path.splitext(os.path.basename(video_path))

    # sensibilites could be changed, performances will vary
    # if image_result = None : no audio or audio too quiet
    image_result , audio_result = predict_video(video_path,
                                            image_model = image_model,
                                            audio_model = audio_model,
                                            image_sensibility=0.9, 
                                            audio_sensibility=0.98,x=file_name+' :')

    if option == 3 :
        return image_result , audio_result

    intersect_result=intersections_audio_image_results(image_result , audio_result)

    if option == 2 :
        return intersect_result

    union_result=combine_audio_image_results(image_result , audio_result)

    if option == 1 :
        return union_result

    if option == 4 :
        return union_result , intersect_result




def predict_directory(directory_path , image_model , audio_model , option = 3 , verbose = True) : 
    # args :
    #     directory_path : path of the directory, will test every file (works only for videos)
    #     image_model : image model.
    #     audio_model : sound model.
    #     option : option for how to combine results 
    #         if 3 : return the two results seperatly image/audio (default choice).
    #         if 4 : return two combined results union/intersection.
    #         otherwise, return a single result.
    #         Imges results are more pertinent, may depend on the video content tho 
    #     verbose : True by default, if False, won't print anything
    #
    # return :
    #     Array of pair of tuple, 
    #         first element is the name of the file, 
    #         second element is the results (depend on option, refer to predict for more informations about results)
    #     If return is None, there is a problem somewhere, should print it exactly.

    start_time = time.time()
    
    if not os.path.exists(directory_path):
        if verbose:
            print(f"Directory does not exist: {directory_path}")
        return []

    predictions = []
    
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.isfile(file_path):
            
            prediction = predict(file_path, image_model, audio_model, option=option, verbose=verbose)
            predictions.append((file_name, prediction))

    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds.")
    return predictions


















def is_video_file(file_path,video_extensions):
    # Check if the file exists
    if not os.path.exists(file_path):
        message = file_path + ": file does not exist"
        return False,message
    
    # Get the file extension
    _, file_extension = os.path.splitext(file_path)
    
    # Check if the file extension is in the list of video extensions
    if file_extension.lower() in video_extensions:
        return True,None
    else:
        message = file_path + ": file exists but is not a supported extension"
        return False,message

def predict_video(video_path, image_model,audio_model,
                             images_interval=1, void_threshold=1, 
                             image_sensibility=0.9, audio_sensibility=0.98,
                             threshold=0.15,x=''):
    start_time = time.time()
    clear_output(wait=True)
    print(x)
    audio_result = predict_audio_of_video(video_path,audio_model,
                                                frame_duration=0.96, 
                                                sensibility=audio_sensibility, 
                                                threshold=threshold, void_threshold=void_threshold, audio_threshold=-50)
    image_result = predict_image_video(video_path, image_model, 
                                                            interval=images_interval, void_threshold=void_threshold,
                                                            sensibility=image_sensibility,
                                                            threshold=threshold)
    elapsed_time = time.time() - start_time
    #print(x)
    print(f"Total time: {elapsed_time:.2f} seconds.")
    return image_result , audio_result


def merge_intervals(intervals1, intervals2):
    # Combine and sort intervals by start time
    intervals = sorted(intervals1 + intervals2, key=lambda x: x[0])
    
    merged = []
    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))
    
    return merged

def intersect_intervals(intervals1, intervals2):
    i, j = 0, 0
    intersection = []
    
    while i < len(intervals1) and j < len(intervals2):
        start1, end1 = intervals1[i]
        start2, end2 = intervals2[j]
        
        # Check if intervals overlap
        if start1 <= end2 and start2 <= end1:
            intersection.append((max(start1, start2), min(end1, end2)))
        
        # Move to the next interval in the list with the earliest ending interval
        if end1 < end2:
            i += 1
        else:
            j += 1
    
    return intersection

def combine_audio_image_results(image_result , audio_result):
    if audio_result is None :
        return image_result
    return merge_intervals(image_result , audio_result)

def intersections_audio_image_results(image_result , audio_result):
    if audio_result is None :
        return image_result
    return intersect_intervals(image_result , audio_result)


