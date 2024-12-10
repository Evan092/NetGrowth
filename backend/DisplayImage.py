import multiprocessing
import os
import random
import threading
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Constants

import cv2
import numpy as np

def resize_and_pad_deprecated(image, desired_size, color=(0, 0, 0)):
    """
    Resize an image to maintain aspect ratio with padding added to make it square.

    Parameters:
    - image: input image as a numpy array.
    - desired_size: the size of the resulting square image (desired_size x desired_size).
    - color: background color for padding, in BGR format.

    Returns:
    - A square image of the specified size.
    """
    old_size = image.shape[:2]  # old_size is in (height, width) format

    # Calculate the ratio of the desired size to the old size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # Resize the image to new_size (maintaining the aspect ratio)
    resized = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)  # new_size should be in (width, height) format

    # Create a new square image and fill with the color
    new_image = np.full((desired_size, desired_size, 3), color, dtype=np.uint8)

    # Compute center offset
    x_offset = (desired_size - new_size[1]) // 2
    y_offset = (desired_size - new_size[0]) // 2

    # Place the resized image at the center of the square canvas
    new_image[y_offset:y_offset + new_size[0], x_offset:x_offset + new_size[1]] = resized

    return new_image

epoch = 0
i = 0
def draw_bounding_boxes(image, truth_boxes, pred_boxes, pred_confidences, epoch1, i1, n, transform=None, truth_color=(255, 0, 0), pred_color=(0, 255, 0), thickness=2):
    """
    Draw truth and predicted bounding boxes on an image and display it.

    Parameters:
    - image_path: path to the image file.
    - truth_boxes: tensor of truth bounding boxes, where each box is [x1, y1, x2, y2].
    - pred_boxes: tensor of predicted bounding boxes, where each box is [x1, y1, x2, y2].
    - truth_color: color of the truth bounding boxes (R, G, B).
    - pred_color: color of the predicted bounding boxes (R, G, B).
    - thickness: line thickness of the boxes.
    """
    epoch=epoch1
    ind=i1
    # Read the image
    
    #image = resize_and_pad(image, Constants.desired_size)

    # Convert colors from BGR to RGB (for Matplotlib compatibility)
    image = np.array(image)

    # Draw truth boxes
    if truth_boxes is not None:
        for (x1, y1, x2, y2) in truth_boxes.squeeze(0):
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
            image = cv2.rectangle(image, start_point, end_point, truth_color, thickness)

    # Draw predicted boxes
    if pred_boxes is not None:
        for i in range(pred_boxes.shape[0]):
            (x1, y1, x2, y2) = pred_boxes[i]
            confidence = pred_confidences[i]
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
            pred_color=(0, 255*pow(confidence.item(),20), 0)
            image = cv2.rectangle(image, start_point, end_point, pred_color, thickness)

    return start_display_process(image, epoch, ind)

def display_image(image):
    """Display an image using matplotlib."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_image(image, file_path):
    """Save an image to a file using matplotlib."""
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(file_path, bbox_inches='tight')  # Save the figure to a file
    plt.close()

def start_display_process(image, epoch, i):
    """Start a new process to display the image."""
    #display_process = multiprocessing.Process(target=display_image, args=(image,))
    folder = "../backend/training_data/verify/epoch " + str(epoch)
    if not os.path.exists(folder):
        os.makedirs(folder)
    save_image(image, folder + "/Image " + str(i))
    return (folder + "/Image " + str(i))
    #display_process.start()
    #display_process.join()  # Wait for the process to close if needed

