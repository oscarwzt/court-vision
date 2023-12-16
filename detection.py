from ultralytics import YOLO
import cv2
import numpy as np
import copy
from mss import mss
import os
import psutil
from tqdm import tqdm
import math
import time
from helpers import is_shot_made, is_ball_in_basket, is_near_basket

def DETECT(model_path, 
           video_path = None, # input video path
           show_output = True, # whether to stream the model detection output to the screen
           save_output = False,
           output_path = "output.mp4",
           verbose = False, # whether to print the model output (objects detected, inference time, etc.)to the console
           show_progress = True,
           jump_to_second = None, # skip to a specific second in the video,
           device = "mps",
           manual_frame = False,
           ):
    
    model = YOLO(model_path)


    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to_second * fps) if jump_to_second is not None else None


    if save_output:
        # Get the dimensions and fps of the video frames
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
    score = 0
    frame_count = 0
    last_score = None
    seconds_to_wait = 3
    frames_to_wait = seconds_to_wait * fps
    prev_ball_center = [0,9999]
    prev_ball = [0,0,0,0]
    score_timestamps = []
    hoop_boxes_cache = []
    

    progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing frames", disable=not show_progress, ncols = 100) if show_progress else None
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        start_time = time.time()

        results = model(img, 
                        stream = True, 
                        device = device, 
                        conf = 0.3, 
                        verbose = verbose)
        for r in results:
            boxes = r.boxes
            hoop_box = (0, 0, 0, 0)
            hoop_box_area = 0
            main_label_index = None  # This will store the index of the largest hoop box
            box_data = []
            ball_box= []
            hoop_boxes = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # hoop
                if model.names[int(box.cls)] == "basketball-hoops":
                    hoop_box_current = (x1, y1, x2, y2)
                    hoop_boxes.append(hoop_box_current)
                    hoop_box_current = (x1, y1, x2, y2)
                    hoop_box_area_cur = (x2 - x1) * (y2 - y1)
                    if hoop_box_area_cur > hoop_box_area:
                        hoop_box = hoop_box_current
                        hoop_box_area = hoop_box_area_cur
                        main_label_index = i  # Store the index of the largest hoop box


                confidence = np.round(box.conf[0].cpu(), 3)
                box_data.append((i, box, x1, y1, x2, y2, confidence))
            # Process box data in a separate loop
            if len(hoop_boxes) >= len(hoop_boxes_cache):
                hoop_boxes_cache = copy.deepcopy(hoop_boxes)
            else:
                hoop_boxes = copy.deepcopy(hoop_boxes_cache)
                
            for i, box, x1, y1, x2, y2, confidence in box_data:
                main_label = i == main_label_index 
                if model.names[int(box.cls)] == "basketball":
                    ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    # print(f"[{i}]:  ", prev_ball_center, '    ', ball_center)
                    cv2.circle(img, ball_center, 2, (0, 255, 0), -1)
                    
                    for hoop in hoop_boxes:
                        if is_near_basket(hoop, ball_center):
                            if is_shot_made(ball_center, prev_ball_center, hoop):
                                if last_score is None or frame_count >= (frames_to_wait + last_score):
                                    print(f"Ball {i}  Score!")
                                    score += 1
                                    last_score = frame_count
                                # Get the timestamp and store it in the list
                                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get timestamp in seconds
                                    score_timestamps.append(timestamp)
                            prev_ball_center = ball_center
                            prev_ball = (x1, y1, x2, y2) if prev_ball != None else prev_ball 
                
                if show_output:    
                    # Draw the ball box
                    if model.names[int(box.cls)] == "basketball-hoops":
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 50, 0), 2)
                        cv2.rectangle(img, (x1-50, y1-50), (x2+50, y2+50), (50, 0, 0), 2)
                    else:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # setup text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_thickness = 2

                    # set the text box color and text color
                    text_box_color = (255, 0, 0) if model.names[int(box.cls)] == "basketball-hoops" else (0, 255, 0)
                    text_color = (255, 255, 255)

                    # the label and confidence values
                    label = f'{model.names[int(box.cls)]} {i} {confidence:.3f}' if not main_label else f'main {model.names[int(box.cls)]} {confidence:.3f}'

                    # get the width and height of the text box
                    (text_width, text_height) = cv2.getTextSize(label, font, fontScale=font_scale, thickness=font_thickness)[0]

                    # set the text box position
                    text_offset_x = x1
                    text_offset_y = y1 - 5

                    # make the coords of the box with a small padding of two pixels
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))

                    cv2.rectangle(img, box_coords[0], box_coords[1], text_box_color, cv2.FILLED)
                    cv2.putText(img, label, (text_offset_x, text_offset_y), font, font_scale, text_color, font_thickness)
        
            frame_count += 1
            
            cv2.putText(img, f'Score: {score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        inference_time = (time.time() - start_time) * 1000

        # Update the progress bar with the new information
        if show_progress:
            progress_bar.set_postfix({
                'inference': f'{inference_time:.1f}ms'
            })
            progress_bar.update(1)

        # Write the processed frame to the video file
        if save_output:
            out.write(img)
        if show_output:
            cv2.imshow('Image', img)
            
            
        
        if cv2.waitKey(1 - int(manual_frame)) & 0xFF == ord('q'):
            progress_bar.close() if show_progress else None
            break
            
    process = psutil.Process(os.getpid())

    # Get the current memory usage in MB
    mem_info = process.memory_info()
    memory_use_GB = mem_info.rss / (1024 ** 3) # rss is the Resident Set Size, and is the portion of the process's memory held in RAM

    print(score_timestamps)
    print(f"Current memory usage: {memory_use_GB} GB")
    
    cap.release()
    out.release() if save_output else None
    cv2.destroyAllWindows()
    
    return score_timestamps
    
 
if __name__ == "__main__":
    DETECT(model_path = "../weights/detect_large.pt", 
           video_path = "/Users/oscarwan/bballDetection/videos/generated_highlights/clip_1.mp4", # input video path
           show_output = True, # whether to stream the model detection output to the screen
           save_output = False,
           output_path = "output.mp4",
           verbose = False, # whether to print the model output (objects detected, inference time, etc.)to the console
           show_progress = True,
           jump_to_second = 5, # skip to a specific second in the video
           manual_frame=False
           )
           