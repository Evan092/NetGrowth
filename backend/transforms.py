import torchvision.transforms.functional as F

import Constants

class ResizeToMaxDimension:
    def __init__(self, max_dim=Constants.desired_size):
        self.max_dim = max_dim

    def __call__(self, image):
        # Get the original image dimensions
        width = 0
        height = 0
        try:
            width, height = image.size
        except Exception as e:
            width, height = image.shape[:2]

        # Determine scaling factor to maintain aspect ratio
        if max(width, height) > self.max_dim:
            scale = self.max_dim / max(width, height)
            new_size = (int(height * scale), int(width * scale))  # Maintain aspect ratio
        else:
            # If the image is already smaller than the max dimension, no resizing
            new_size = ( height,width)

        # Resize the image while preserving the aspect ratio
        return F.resize(image, new_size)