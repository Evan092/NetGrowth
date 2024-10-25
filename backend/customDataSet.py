import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

class CustomImageDataset(Dataset):
    maxHeight = 0
    maxWidth = 0
    maxBBoxes = 0
    maxValid = 0

    def __init__(self, img_dir, transform=None, train=False):
        self.img_dir = img_dir
        self.transform = transform
        self.train = train
        self.image_filenames = [f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]

        # Set the appropriate JSON file based on training or test data
        if train:
            annotations_file = "./backend/training_data/TextOCR_0.1_train.json"
            #csv_file = "./backend/training_data/train-images-boxable-with-rotation.csv"
        else:
            annotations_file = "./backend/training_data/TextOCR_0.1_val.json"
            #csv_file = "./backend/training_data/train-images-boxable-with-rotation.csv"

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
        _, width, height = image_tensor.shape
        
        # Calculate padding needed for both width and height
        pad_width = max(0, target_width - width)
        pad_height = max(0, target_height - height)
        
        # Padding is applied as (top, right, bottom, left)
        #padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
        padding = (pad_height // 2, pad_width // 2, pad_height - pad_height // 2, pad_width - pad_width // 2)
        
        # Apply padding using F.pad (for tensors), padding must be in (left, right, top, bottom) order
        padded_image_tensor = F.pad(image_tensor, padding, fill=0)
        
        return padded_image_tensor, (pad_width // 2), (pad_height // 2)

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

        if self.transform:
            oldSize = image.size
            image = self.transform(image)
            newSize = (image.shape[1], image.shape[2])
            scale_x, scale_y = self.getScales(oldSize, newSize)
            image, adjustX, adjustY  = self.pad_to_target_size(image, self.maxWidth, self.maxHeight)

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
                    bbox[0] = (bbox[0]*scale_x) + adjustX
                    #additionalAdjust = (bbox[1]*scale_y)/2
                    bbox[1] = (bbox[1]*scale_y) + adjustY
                    bbox[2] = (bbox[2]*scale_x) + adjustX
                    bbox[3] = (bbox[3]*scale_y) + adjustY


                    # Append bounding box and the length of the string
                    #if ann['utf8_string'] != ".":
                    bboxes_with_lengths.append((bbox, len(ann['utf8_string'])))

            if len(bboxes_with_lengths) > self.maxValid:
                self.maxValid = max(self.maxValid, len(bboxes_with_lengths))

            # Sort bounding boxes by the length of the utf8_string in descending order
            bboxes_with_lengths.sort(key=lambda x: x[1], reverse=True)

            # Extract just the bounding boxes for the top maxBBoxes entries
            bboxes = [bbox for bbox, _ in bboxes_with_lengths[:self.maxBBoxes]]

            if len(bboxes) == 0:
                print("None")

            while len(bboxes) < self.maxBBoxes:
                bboxes.append([0,0,0,0])

            # Convert bounding boxes to tensor
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

            # Return image and bounding boxes as tensors
            return image, bboxes, img_path
        else:
            return image
