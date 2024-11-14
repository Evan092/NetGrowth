import json
import math
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import pandas as pd
import Constants
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomImageDataset(Dataset):

    def __init__(self, img_dir, transform=None, train=False, anchor_boxes=None):
        self.maxHeight = Constants.desired_size
        self.maxWidth = Constants.desired_size
        self.maxBBoxes = Constants.max_boxes
        self.maxValid = Constants.max_boxes
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        self.image_filenames = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]
        self.anchor_boxes = anchor_boxes
        self.rotation_transform = RandomRotationWithBBox(angle_range=(-10, 10), p=0.5)

        # Set the appropriate JSON file based on training or test data
        if train:
            annotations_file = "./backend/training_data/TextOCR_0.1_train.json"
            self.csv_file = "./backend/training_data/train-images-boxable-with-rotation.csv"
        else:
            annotations_file = "./backend/training_data/TextOCR_0.1_val.json"
            self.csv_file = "./backend/training_data/train-images-boxable-with-rotation.csv"

        # Load JSON data (images and optionally annotations)
        with open(annotations_file, 'r') as f:
            self.data = json.load(f)

        # Extract image metadata
        self.imgs = self.data['imgs']
        
        # Only load annotations if it's training data
        if True:
            self.anns = self.data.get('anns', {})  # Safely get annotations, default to empty if not found
            self.img2Anns = self.data.get('imgToAnns', {})  # Image to annotations mapping

        # Create a list of image IDs for iteration
        self.image_ids = list(self.imgs.keys())
        #self.setMaxHeight()
        #self.setMaxWidth()
        self.df = pd.read_csv(self.csv_file)

    def setMaxHeight(self):
        for image in self.imgs:
            if self.imgs[image]["height"] > self.maxHeight:
                self.maxHeight = self.imgs[image]["height"]


    def setMaxWidth(self):
        for image in self.imgs:
            if self.imgs[image]["width"] > self.maxWidth:
                self.maxWidth = self.imgs[image]["width"]

    def overwriteMaxWidth(self, newWidth):
        self.maxWidth = max(self.maxWidth, newWidth)

    def overwriteMaxHeight(self, newHeight):
        self.maxHeight = max(self.maxHeight, newHeight)

    def setMaxDimensions(self, height, width):
        self.overwriteMaxWidth(width)
        self.overwriteMaxHeight(height)

    def __len__(self):
        return len(self.image_ids)  # This should return 21,778, not 3

    def pad_bboxes(bboxes, max_boxes):
        padded_bboxes = torch.zeros((max_boxes, 4))  # Create a tensor of max_boxes with 4 coordinates each (all zeroes)
        padded_bboxes[:bboxes.shape[0], :] = bboxes  # Copy existing boxes
        return padded_bboxes

    def setMaxBBoxes(self, maxBBoxes):
        self.maxBBoxes = maxBBoxes

    def pad_to_target_size(self, image_tensor, target_width, target_height):
        # Get the current tensor dimensions (assuming shape is [C, W, H])
        _, height, width = image_tensor.shape
        
        # Calculate padding needed for both width and height
        pad_width = max(0, target_width - width)
        pad_height = max(0, target_height - height)
        
        # Padding is applied as (top, right, bottom, left)
        PadLeft = random.randint(0, pad_width)
        PadRight = pad_width - PadLeft
        PadTop = random.randint(0, pad_height)
        PadBottom = pad_height - PadTop



        #padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
        #padding = (pad_height // 2, pad_width // 2, pad_height - pad_height // 2, pad_width - pad_width // 2)
        padding = (PadLeft, PadTop, PadRight, PadBottom)
        
        # Apply padding using F.pad (for tensors), padding must be in (left, right, top, bottom) order
        padded_image_tensor = F.pad(image_tensor, padding, fill=0)


        return padded_image_tensor, padding

    def getScales(self, original_size, new_size):
        old_width, old_height = original_size
        new_width, new_height = new_size
        
        # Scale factors for resizing
        scale_x = new_width / old_width
        scale_y = new_height / old_height
        
        return scale_x, scale_y

    def points_to_bbox(self, points):
        """
        Convert a list of points defining a polygon into a bounding box.

        Parameters:
        - points: List of floats where each pair of floats represents x, y coordinates of a vertex.

        Returns:
        - bbox: A list containing [min_x, min_y, max_x, max_y] which defines the bounding box.
        """
        x_coordinates = points[0::2]  # Extract all x coordinates
        y_coordinates = points[1::2]  # Extract all y coordinates

        min_x = min(x_coordinates)
        max_x = max(x_coordinates)
        min_y = min(y_coordinates)
        max_y = max(y_coordinates)

        bbox = [min_x, min_y, max_x, max_y]
        return bbox


    def __getitem__(self, idx):
        try:
            image_id = self.image_ids[idx]
            img_data = self.imgs[image_id]
            img_path = os.path.join(self.img_dir, img_data['file_name'])
            # Load image
            image = Image.open(img_path)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            adjustX = 0
            adjustY=0
            scale_x=1
            scale_y=1
            angle = 0
            angle1=0

            if self.transform:

                # Iterate through the file in chunks
                row = self.df[self.df.iloc[:, 0] == image_id]
                if not row.empty:
                    rotation_value = row["Rotation"].values[0]

                if math.isnan(rotation_value):
                    rotation_value = 0.0
                    

                oldSize = (image.height, image.width)
                image = self.transform(image)
                newSize = (image.shape[1], image.shape[2])
                scale_y, scale_x  = self.getScales(oldSize, newSize)
                image, (adjustX1, adjustY1, adjustX2, adjustY2)  = self.pad_to_target_size(image, self.maxWidth, self.maxHeight)
                image, angle1 = self.rotation_transform(image, rotation_value)

                image, angle = self.rotation_transform(image)

            if True:
                # Retrieve the annotations for this image
                ann_ids = self.img2Anns.get(image_id, [])
                bboxes = []
                bboxes_with_lengths = []
                for ann_id in ann_ids:
                    ann = self.anns.get(ann_id)
                    if ann:
                        bbox = self.points_to_bbox(ann['points'])  # Extract bounding box
                        if bbox[0] > bbox[2]:
                            temp = bbox[2]
                            bbox[2] = bbox[0]
                            bbox[0] = temp

                        if bbox[1] > bbox[3]:
                            temp = bbox[3]
                            bbox[3] = bbox[1]
                            bbox[1] = temp

                        

                        #additionalAdjust = (bbox[0]*scale_x)/2
                        bbox[0] = (bbox[0]*scale_x) + adjustX1
                        #additionalAdjust = (bbox[1]*scale_y)/2
                        bbox[1] = (bbox[1]*scale_y) + adjustY1
                        bbox[2] = (bbox[2]*scale_x) + adjustX1
                        bbox[3] = (bbox[3]*scale_y) + adjustY1

                        bbox = RandomRotationWithBBox.rotateBBox(bbox, -(angle+angle1))
                        # Append bounding box and the length of the string
                        #if ann['utf8_string'] != ".":
                        bboxes_with_lengths.append((bbox, len(ann['utf8_string'])))

                #if len(bboxes_with_lengths) > self.maxValid:
                    #self.maxValid = max(self.maxValid, len(bboxes_with_lengths))

                # Sort bounding boxes by the length of the utf8_string in descending order
                bboxes_with_lengths.sort(key=lambda x: x[1], reverse=True)

                # Extract just the bounding boxes for the top maxBBoxes entries
                bboxes = [bbox for bbox, _ in bboxes_with_lengths[:self.maxBBoxes]]

                if len(bboxes) == 0:
                    print("None")

                #while len(bboxes) < self.maxBBoxes:
                    #bboxes.append([0,0,0,0])

                # Convert bounding boxes to tensor
                bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

                small_anchors = self.anchor_boxes[:3]    # First 3 anchors for small scale
                medium_anchors = self.anchor_boxes[3:6]  # Next 3 anchors for medium scale
                large_anchors = self.anchor_boxes[6:9]   # Last 3 anchors for large scale

                small_scale_boxes = []
                medium_scale_boxes = []
                large_scale_boxes = []

                # Function to determine the closest scale based on anchor boxes
                def get_closest_scale(width, height):
                    box_size = torch.tensor([width, height], dtype=torch.float32)

                    # Calculate differences for each scale and find the minimum
                    small_diff = torch.min(torch.norm((small_anchors*Constants.desired_size) - box_size, dim=1))
                    medium_diff = torch.min(torch.norm((medium_anchors*Constants.desired_size) - box_size, dim=1))
                    large_diff = torch.min(torch.norm((large_anchors*Constants.desired_size) - box_size, dim=1))

                    # Select the scale with the smallest difference
                    min_diff, scale = torch.min(torch.tensor([small_diff, medium_diff, large_diff]), dim=0)
                    return scale.item()  # 0 for small, 1 for medium, 2 for large

                # Assign each bounding box to the closest scale
                for box in bboxes:
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1

                    # Determine the closest scale
                    scale = get_closest_scale(width, height)
                    if scale == 0:
                        small_scale_boxes.append(box)
                    elif scale == 1:
                        medium_scale_boxes.append(box)
                    else:
                        large_scale_boxes.append(box)

                # Convert lists to tensors
                small_scale_boxes = torch.stack(small_scale_boxes) if small_scale_boxes else torch.empty((0, 4), dtype=torch.float32)
                medium_scale_boxes = torch.stack(medium_scale_boxes) if medium_scale_boxes else torch.empty((0, 4), dtype=torch.float32)
                large_scale_boxes = torch.stack(large_scale_boxes) if large_scale_boxes else torch.empty((0, 4), dtype=torch.float32)

                all_bboxes = [small_scale_boxes, medium_scale_boxes, large_scale_boxes]
                # Return image and bounding boxes as tensors
                return image, all_bboxes, img_path
            else:
                return image
        except:
            print("reee")
        
class RandomRotationWithBBox:
    def __init__(self, angle_range=(-10, 10), p=0.5):
        self.angle_range = angle_range
        self.p = p

    def __call__(self, img_tensor, angle=None):
        if angle is None:
            angle = random.uniform(*self.angle_range) if random.random() < self.p else 0
        if angle == 0:
            return img_tensor, angle
        img_tensor = F.rotate(img_tensor, angle)
        return img_tensor, angle

    @staticmethod
    def rotateBBox(box, angle):
        if angle == 0:
            return box  # No rotation needed

        w, h = Constants.desired_size, Constants.desired_size
        x_min, y_min, x_max, y_max = box

        # Step 2: Define the center of the image
        cx, cy = Constants.desired_size / 2, Constants.desired_size / 2

        # Step 3: Define the four corners of the bounding box
        corners = [
            (x_min, y_min),  # top-left
            (x_max, y_min),  # top-right
            (x_min, y_max),  # bottom-left
            (x_max, y_max)   # bottom-right
        ]

        # Step 4: Convert the angle to radians and calculate cosine and sine
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Step 5: Rotate each corner around the image center
        rotated_corners = []
        for x, y in corners:
            # Translate corner to image center
            x_shifted, y_shifted = x - cx, y - cy

            # Apply rotation matrix
            new_x = x_shifted * cos_a - y_shifted * sin_a + cx
            new_y = x_shifted * sin_a + y_shifted * cos_a + cy
            rotated_corners.append((new_x, new_y))

        # Step 6: Determine the new bounding box coordinates
        new_x_min = min(c[0] for c in rotated_corners)
        new_y_min = min(c[1] for c in rotated_corners)
        new_x_max = max(c[0] for c in rotated_corners)
        new_y_max = max(c[1] for c in rotated_corners)

        # Output the final rotated bounding box
        rotated_bbox = [new_x_min, new_y_min, new_x_max, new_y_max]

        return rotated_bbox