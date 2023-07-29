from ultralytics import YOLO
import cv2
import numpy as np
from mss import mss
import os
import psutil
from tqdm import tqdm
import math
import time

def DETECT(model_path, 
           is_screen_capture,
           video_path = None, 
           show_output = True,
           save_output = False,
           output_path = "output.mp4",
           verbose = False,
           show_progress = True,
           jump_to_second = None):
    
    model = YOLO(model_path)

    if not is_screen_capture:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, jump_to_second * fps) if jump_to_second is not None else None
    else:
        sct = mss()

    if save_output:
        # Get the dimensions and fps of the video frames
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
    score = 0
    frames_after_score = None
    seconds_to_wait = 3
    frames_to_wait = seconds_to_wait * fps
    prev_ball_center = None
    score_timestamps = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Processing frames", disable=not show_progress, ncols = 100) if show_progress else None
    
    while True:
        if not is_screen_capture:
            success, img = cap.read()
            if not success:
                break
        else:
            # Capture computer screen
            screenshot = sct.grab(sct.monitors[1])
            
            # Convert the screenshot to a numpy array and then to a color image
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        frames_processed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        start_time = time.time()

        results = model(img, 
                        stream = True, 
                        device = "mps", 
                        conf = 0.4, 
                        verbose = verbose)
        
        for r in results:
            print("\n =========== \n")
            boxes = r.boxes
            hoop_box = (0, 0, 0, 0)
            hoop_box_area = 0
            main_label_index = None  # This will store the index of the largest hoop box
            box_data = []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # hoop
                if model.names[int(box.cls)] == "basketball-hoops":
                    hoop_box_current = (x1, y1, x2, y2)
                    hoop_box_area_cur = (x2 - x1) * (y2 - y1)
                    if hoop_box_area_cur > hoop_box_area:
                        hoop_box = hoop_box_current
                        hoop_box_area = hoop_box_area_cur
                        main_label_index = i  # Store the index of the largest hoop box

                confidence = np.round(box.conf[0].cpu(), 3)
                box_data.append((i, box, x1, y1, x2, y2, confidence))

            # Process box data in a separate loop
            for i, box, x1, y1, x2, y2, confidence in box_data:
                
                main_label = i == main_label_index 
                if model.names[int(box.cls)] == "basketball":
                    ball_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    print(f"[{i}]:  ", prev_ball_center, '    ', ball_center)
                    cv2.circle(img, ball_center, 2, (0, 255, 0), -1)
                    
                    hoop_x1, hoop_y1, hoop_x2, hoop_y2 = hoop_box
                    # check if a bucket is made by checking"
                    # whether the ball is in the hoop and 
                    # whether the previous ball position > current
                    ball_is_in_hoop = hoop_x1 <= ball_center[0] <= hoop_x2 and hoop_y1 <= ball_center[1] <= hoop_y2
                    ball_lower_than_previous = prev_ball_center is not None and prev_ball_center[1] > ball_center[1]
                    ball_above_hoop_previously = prev_ball_center is not None and prev_ball_center[1] > hoop_y1
                    if ball_is_in_hoop and ball_lower_than_previous and ball_above_hoop_previously:
                        print(f"Ball {i}  Score!")
                        print(f"Previous ball center: {prev_ball_center}")
                        print(f"Current ball center: {ball_center}")
                        print(f"ball is in hoop: {ball_is_in_hoop}")
                        print(f"ball lower than previous: {ball_lower_than_previous}")
                        if frames_after_score is None or frames_after_score >= frames_to_wait:
                            
                            score += 1
                            frames_after_score = 0  
                            
                            # Get the timestamp and store it in the list
                            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Get timestamp in seconds
                            score_timestamps.append(timestamp)
                    prev_ball_center = ball_center 
                
                if not show_output: continue    

                # Draw the ball box
                if model.names[int(box.cls)] == "basketball-hoops":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
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
        
            if frames_after_score is not None:
                frames_after_score += 1 
            
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
            
            
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
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

def TRACK(model_path, video_path):
    model = YOLO(model_path)
    model.track(source = video_path,)
    
def generateHighlight(video_path,
                      score_timestamps, 
                      clip_start_offset = 6, # number of seconds before scoring
                      clip_end_offset = 3,   # number of seconds after scoring
                      output_path = "/Users/oscarwan/bballDetection/videos_clipped/scored"):
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Video codec
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate clip lengths in frames
    start_frame_offset = clip_start_offset * fps
    end_frame_offset = clip_end_offset * fps

    # For each score event
    for i, timestamp in enumerate(score_timestamps):
        print(f'Processing clip {i}')
        # Calculate start and end frames for this clip
        score_frame = math.floor(timestamp * fps)
        start_frame = int(max(0, score_frame - start_frame_offset))
        end_frame = min(total_frames - 1, math.ceil(score_frame + end_frame_offset))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        video_path = f'{output_path}/clip_{i}.mp4'
        out = cv2.VideoWriter(video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        # Copy frames from the input video to the output clip
        for _ in tqdm(range(start_frame, end_frame + 1)):

            ret, frame = cap.read()

            if ret:
                out.write(frame)
            else:
                break

        # Close the output clip
        out.release()

    # Close the input video
    cap.release()
    
if __name__ == "__main__":
    
    video_path = 'videos_clipped/scored/clip_9_x.mp4'
    
    #stamps = [28.328300000000002, 35.101733333333335, 54.554500000000004, 63.09636666666667, 80.9809, 90.85743333333333, 112.11200000000001, 135.16836666666669, 168.5684, 176.04253333333335, 213.37983333333335, 223.65676666666667, 246.57966666666667, 263.3964666666667, 266.63303333333334, 289.4224666666667, 300.73376666666667, 305.17153333333334, 311.97833333333335, 340.9406, 358.5582, 361.56120000000004, 380.9138666666667, 384.1838, 390.99060000000003, 411.37763333333334, 432.3319, 444.21043333333336, 462.2284333333333, 485.6184666666667, 498.8650333333334, 511.87803333333335, 521.3208000000001, 586.2857, 593.3260666666667]
    stamps = DETECT(model_path = 'weights/v10_120_s.pt', 
       is_screen_capture=False,
       show_output = True, 
       video_path = video_path,
       save_output = False,
       #output_path="v9.mp4",
       verbose = False,
       show_progress = False,
       jump_to_second = 5
       )
    
    model = YOLO('weights/v10_120_s.pt')
    model.predict(source = video_path, show = True, device = "mps")


