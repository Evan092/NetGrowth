import json
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import os

from BoundingBoxCNN import *

def points_to_bbox(points):
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

def load_and_prepare_bboxes(annotations_file, img_dir, transform=None):
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    bounding_boxes = []
    for image_id, img_data in data['imgs'].items():
        img_path = os.path.join(img_dir, img_data['file_name'])
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        oldSize = image.size
        scale_x, scale_y = 1, 1  # Default scale factors
        adjustX, adjustY = 0, 0  # Default adjustments
        
        if transform:
            image = transform(image)
            newSize = (image.shape[1], image.shape[2])
            scale_x = newSize[0] / oldSize[0]
            scale_y = newSize[1] / oldSize[1]
        
        ann_ids = data['imgToAnns'].get(image_id, [])
        for ann_id in ann_ids:
            ann = data['anns'][ann_id]
            if ann:
                bbox = points_to_bbox(ann['points'])  # Extract bounding box
                if bbox[0] > bbox[2]:
                    temp = bbox[2]
                    bbox[2] = bbox[0]
                    bbox[0] = temp

                if bbox[1] > bbox[3]:
                    temp = bbox[3]
                    bbox[3] = bbox[1]
                    bbox[1] = temp

                x_min = (bbox[0] * scale_x) + adjustX
                y_min = (bbox[1] * scale_y) + adjustY
                x_max = (bbox[2] * scale_x) + adjustX
                y_max = (bbox[3] * scale_y) + adjustY

                # Normalize dimensions
                norm_width = (x_max - x_min) / newSize[0]
                norm_height = (y_max - y_min) / newSize[1]
                bounding_boxes.append([norm_width, norm_height])

    return np.array(bounding_boxes)

def perform_kmeans_clustering(bounding_boxes, k=9):
    """ Perform k-means clustering on normalized bounding box dimensions. """
    kmeans = KMeans(n_clusters=k, random_state=42).fit(bounding_boxes)
    return kmeans.cluster_centers_

def perform_elbow_method(bounding_boxes):
    distortions = []
    K = range(1, 12)  # Testing from 1 to 11 clusters
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(bounding_boxes)
        distortions.append(kmeanModel.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

# Parameters
annotations_path = "./backend/training_data/TextOCR_0.1_train.json"
images_directory = "./backend/training_data/"
k_clusters = 1  # Adjust as needed




# Assuming 'transform' is defined elsewhere in your pipeline
data_for_clustering = load_and_prepare_bboxes(annotations_path, images_directory, transform=transform)
#anchor_boxes = perform_kmeans_clustering(data_for_clustering, k_clusters)
#print("Determined Anchor Boxes:", anchor_boxes)

# Assuming 'bounding_boxes' is your array of bounding box dimensions
kmeans = KMeans(n_clusters=1, random_state=42)
kmeans.fit(data_for_clustering)

# Convert NumPy array of cluster centers to a list
anchor_boxes = kmeans.cluster_centers_.tolist()

anchor_boxes = sorted(anchor_boxes, key=lambda x: x[0] * x[1])

# Save the anchor boxes to a JSON file
with open('anchor_boxes.json', 'w') as file:
    json.dump(anchor_boxes, file, indent=4)

#perform_elbow_method(data_for_clustering)
