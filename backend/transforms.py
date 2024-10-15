import torchvision.transforms.functional as F

class ResizeToMaxDimension:
    def __init__(self, max_dim=1024):
        self.max_dim = max_dim

    def __call__(self, image):
        # Get the original image dimensions
        width, height = image.size

        # Determine scaling factor to maintain aspect ratio
        if max(width, height) > self.max_dim:
            scale = self.max_dim / max(width, height)
            new_size = (int(width * scale), int(height * scale))  # Maintain aspect ratio
        else:
            # If the image is already smaller than the max dimension, no resizing
            new_size = (width, height)

        # Resize the image while preserving the aspect ratio
        return F.resize(image, new_size)