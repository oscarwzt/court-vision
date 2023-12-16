import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
import datetime
import numpy as np
import json
import cv2
from IPython.display import HTML
from base64 import b64encode
import os
from IPython.display import Video
import subprocess

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10, chkpt_interval = 5, checkpoint_dir='./cls_chkpoint', early_stopping_patience=3):
    best_accuracy = 0.0  # Initialize with a threshold accuracy
    best_loss = float('inf')
    early_stopping_counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    model.to(device)
    
    now = datetime.datetime.now()
    checkpoint_dir = os.path.join(checkpoint_dir, now.strftime('checkpoint_%Y-%m-%d-%H-%M'))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Initialize arrays to track the losses and accuracies
    losses = np.empty(num_epochs)
    test_losses = np.empty(num_epochs)
    train_accuracies = np.empty(num_epochs)
    test_accuracies = np.empty(num_epochs)
    
    for epoch in range(num_epochs):
        temp = time.time()
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        
        # Early stopping check
        if test_losses[epoch] < best_loss:
            best_loss = test_losses[epoch]
            early_stopping_counter = 0  # reset the early stopping counter if the validation loss improves
        else:
            early_stopping_counter += 1  # increment the counter if the validation loss does not improve
            
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Validation loss did not improve for {early_stopping_patience} consecutive epochs.")
            break
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        losses[epoch] = avg_loss
        train_accuracies[epoch] = correct_predictions / len(train_loader.dataset)
        
        model.eval()
        correct_test_predictions = 0
        total_test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                test_loss = criterion(outputs, targets)
                total_test_loss += test_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_test_predictions += (predicted == targets).sum().item()
        
        test_accuracies[epoch] = correct_test_predictions / len(test_loader.dataset)
        test_losses[epoch] = total_test_loss / len(test_loader.dataset)
        
        # Save checkpoint after every epoch
        if (epoch + 1) % chkpt_interval == 0:
            checkpoint = {
               'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': losses[:epoch+1],
                'train_acc': train_accuracies[:epoch+1],
                'val_loss': test_losses[:epoch+1],
                'val_acc': test_accuracies[:epoch+1]
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
        
        # Save the best model if test accuracy has improved
        if test_accuracies[epoch] > best_accuracy:
            best_accuracy = test_accuracies[epoch]
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
        
        # Print statistics
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]\t Avg Loss: {avg_loss:.4f}\t "
                  f"Val Loss: {test_losses[epoch]:.4f}\t"
                  f"TrainAcc: {train_accuracies[epoch]:.4f} \t TestAcc: {test_accuracies[epoch]:.4f}\t "
                  f"Time: {(time.time() - temp):.4f}")
    
        history = {
            'train_loss': losses[:epoch+1].tolist(),
            'train_acc': train_accuracies[:epoch+1].tolist(),
            'val_loss': test_losses[:epoch+1].tolist(),
            'val_acc': test_accuracies[:epoch+1].tolist()
        }
        # save history to checkpoint_dir as json
        with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(history, f)
    
    return history


    
def plot_training_history(history):

    plt.figure(figsize=(12, 4))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()
    
def load_checkpoint(model, checkpoint_path, num_classes = 2, ):
    """
    Load a model checkpoint and restore the model and optimizer states.

    Parameters:
    - model: The model instance on which to load the parameters.
    - optimizer: The optimizer instance for which to restore the state.
    - checkpoint_path: The path to the checkpoint file.

    Returns:
    - model: The model with restored parameters.
    - optimizer: The optimizer with restored state.
    - epoch: The epoch at which the checkpoint was saved.
    - loss: The loss value at the checkpoint.
    """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    
def cls_predict_image(cls_model, img, preprocess, device, threshold = 0.5):
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        cls_output = cls_model(input_tensor)
    probability = torch.nn.functional.softmax(cls_output[0], dim=0)

    # prob, predicted_class = torch.max(probability, dim=0)
    # return predicted_class.item(), prob.item()
    return probability[1] > threshold, probability[1].item()

def predict_hoop_box(img, cls_model, x1, y1, x2, y2, preprocess, device, grayscale = False, threshold = 0.5):
    cropped_img = img[y1:y2, x1:x2]
    cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)) if not grayscale else Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY))
    # Preprocess the cropped image
    predicted_class, prob = cls_predict_image(cls_model, cropped_img_pil, preprocess, device)
    return cropped_img_pil, predicted_class, prob

def cls_predict_batch(cls_model, batch_imgs, preprocess, device, threshold = 0.5):
    # Process and batch images
    batch_tensor = torch.stack([preprocess(img).to(device) for img in batch_imgs])

    # Forward pass for the whole batch
    with torch.no_grad():
        cls_output = cls_model(batch_tensor)

    # Calculate probabilities and predicted classes
    probabilities = torch.nn.functional.softmax(cls_output, dim=1)
    return probabilities[:, 1] > threshold, probabilities[:, 1].tolist()

def prepare_hoop_box_batch(img, cls_model,  preprocess, device, grayscale=False, threshold = 0.5):
    cropped_imgs_pil = []

    for cropped_img in img:
        if grayscale:
            cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY))
        else:
            cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
        cropped_imgs_pil.append(cropped_img_pil)

    return cls_predict_batch(cls_model, cropped_imgs_pil, preprocess, device, threshold)



def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        raise ValueError("Error: Could not open the video file.")
    
    # Get the frame width and frame height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = cap.get(cv2.CAP_PROP_FPS)

    return cap, fps, frame_width, frame_height

def display_video(video_path):
    return HTML(f"""
        <video alt="test" controls width="640" height="360">
            <source src="{video_path}" type="video/mp4">
        </video>
    """)

def display_compressed_video(input_path):
    output_path = "compressed_" + input_path
    os.remove(output_path) if os.path.exists(output_path) else None
    try:
        # Use subprocess to safely call FFmpeg
        subprocess.run(['ffmpeg', '-i', input_path, '-vcodec', 'libx264', output_path], check=True)

        # Read and encode the compressed video
        with open(output_path, 'rb') as file:
            mp4 = file.read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()

        # Display video in HTML
        display_html = f"""
        <video width=800 controls>
            <source src="{data_url}" type="video/mp4">
        </video>
        """
        return HTML(display_html)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def generateHighlight(video_path,
                      score_timestamps, 
                      clip_start_offset = 6, # number of seconds before scoring
                      clip_end_offset = 3,   # number of seconds after scoring
                      output_path = "videos_clipped/scored"):
    
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
        score_frame = np.floor(timestamp * fps)
        start_frame = int(max(0, score_frame - start_frame_offset))
        end_frame = min(total_frames - 1, np.ceil(score_frame + end_frame_offset))
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