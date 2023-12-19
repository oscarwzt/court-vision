import cv2
import torch
from ultralytics import YOLO
import numpy as np
import math
from numpy import random
from IPython.display import HTML
import torchvision.models as torch_models
from base64 import b64encode
import os
from IPython.display import Video
from utils import *
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import display
import argparse

preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

yolo_model_path = "weights/detect_large.pt"
model = YOLO(yolo_model_path)
classNames = ['basketball', 'hoop', 'person']

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

def inference_by_batch(yolo_model_path,
                       cls_model_path,
                       video_path, 
                       cls_conf_threshold = 0.6,
                       detect_conf_threshold = 0.4,
                       save_result_vid = False, 
                       output_dir = None, 
                       saved_video_name = None,
                       batch_size=128,
                       display_result = False,
                       show_progress = True,
                       skip_to_sec = 0,
                       show_score_prob = False,
                       ):
    
    model = YOLO(yolo_model_path)
    cls_model = load_resnet50(cls_model_path, device=device)
    
    cap, fps, frame_width, frame_height = get_video_info(video_path)
    if skip_to_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, skip_to_sec * 1000)
        
    num_skiped_frames = int(skip_to_sec * fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - num_skiped_frames
    
    
    if save_result_vid:
        video_name = video_path.split("/")[-1]
        video_name = video_name.split(".")[0] + ".mp4"
        output_path = "inferenced_" + video_name if output_dir is None else os.path.join(output_dir, "inferenced_" + video_name)
        if saved_video_name is not None:
            compressed_output_path = saved_video_name if output_dir is None else os.path.join(output_dir, saved_video_name)
        else:
            compressed_output_path = "compressed_inferenced_" + video_name if output_dir is None else os.path.join(output_dir, "compressed_inferenced_" + video_name)
        codec = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, codec, fps, (frame_width,frame_height))
    
    num_batches = math.ceil(total_frames / batch_size)

    results = []
    score_timestamps = []
    
    count = 0
    score = 0
    display_prob = [0.0]
    
    if show_progress:
        batch_range = tqdm(range(num_batches))
    else:
        batch_range = range(num_batches)

    for i in batch_range:
        frames = []
        for i in range(batch_size):
            ret, img = cap.read()
            if ret:
                frames.append(img)
            else:
                break

        if frames:
            results = model(frames, 
                            stream=False, 
                            verbose = False, 
                            conf=detect_conf_threshold)
        else:
            continue

        for c, r in enumerate(results):
            img = r.orig_img
            boxes = r.boxes
            cropped_images = []
            count += 1
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # convert to int values
                confidence = box.conf.item()
                predicted_class = model.names[int(box.cls)] 
                if predicted_class == "hoop":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    
                    if x1 > x2 or y1 > y2:
                        continue
                    else:
                        cropped_img = img[y1:y2, x1:x2]
                        cropped_images.append(cropped_img)
                        
                if predicted_class == "basketball":
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'{predicted_class}: {confidence:.3f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
            
            if len(cropped_images) == 0:
                continue
            pred, prob = predict_hoop_box_batch(cropped_images, cls_model,  preprocess, device, threshold=cls_conf_threshold)
            if pred.sum() > 0 and count > 60:
                score += 1
                count = 0
                current_frame = i * batch_size + c
                time_stamp = current_frame / fps
                score_timestamps.append((time_stamp, prob))
                display_prob = prob
        
            cv2.putText(img, f'Score: {score}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            if show_score_prob:
                cv2.putText(img, f'Prob: {max(display_prob):.3f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            if save_result_vid:
                out.write(img)
        if not ret:
            break
        
    if save_result_vid:
        out.release()
    cap.release()
    
    if save_result_vid:
        subprocess.run(['ffmpeg', '-y', '-hide_banner',  '-loglevel', 'error', '-i', output_path, '-vcodec', 'libx264', compressed_output_path], check=False)
        os.remove(output_path)
        if display_result:
            display(Video(compressed_output_path, embed=True))
        return score_timestamps, compressed_output_path
    else:
        return score_timestamps


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run inference on a video using a model.')

    # Add arguments
    parser.add_argument('yolo_model_path', help='Path to the model file')
    parser.add_argument('cls_model_path', help='Path to the classification model file')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--cls_conf_threshold', type=float, default=0.6, help='Classification confidence threshold')
    parser.add_argument('--detect_conf_threshold', type=float, default=0.4, help='Detection confidence threshold')
    parser.add_argument('--save_result_vid', action='store_true', help='Flag to save result video')
    parser.add_argument('--output_dir', default=None, help='Output directory for saved results')
    parser.add_argument('--saved_video_name', default=None, help='Name for the saved video')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for processing')
    parser.add_argument('--display_result', action='store_true', help='Flag to display result')
    parser.add_argument('--show_progress', action='store_true', help='Flag to show progress')
    parser.add_argument('--skip_to_sec', type=int, default=0, help='Seconds to skip to in the video')
    parser.add_argument('--show_score_prob', action='store_true', help='Flag to show score probability')

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    inference_by_batch(args.yolo_model_path, 
                       args.cls_model_path,
                       args.video_path, 
                       cls_conf_threshold=args.cls_conf_threshold,
                       detect_conf_threshold=args.detect_conf_threshold,
                       save_result_vid=args.save_result_vid, 
                       output_dir=args.output_dir, 
                       saved_video_name=args.saved_video_name,
                       batch_size=args.batch_size,
                       display_result=args.display_result,
                       show_progress=args.show_progress,
                       skip_to_sec=args.skip_to_sec,
                       show_score_prob=args.show_score_prob)

if __name__ == "__main__":
    main()