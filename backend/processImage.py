from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

def grayScale(img):
    #Read the image
    #convert is to grayscale
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return im_gray

def getThreshhold(im_gray):
    #find the threshhold. Anything that's less than X intensity set to 0, more than X set to 255
    ret, im_th = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret, im_th

def createBoundingByLetter(im_th, gap_factor=2.0):
    ctrs, hier = cv2.findContours(im_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [cv2.boundingRect(c) for c in ctrs]

    # Filter out bounding boxes with width >= 10 and height >= 10
    filtered_bboxes = [b for b in bboxes if b[2] >= 10 and b[3] >= 10]

    # Sort the filtered bounding boxes by their y and x coordinates
    sorted_bboxes = sorted(filtered_bboxes, key=lambda b: (b[1], b[0]))


     # Compute average spacing between adjacent bounding boxes in the same row
    average_spacing = 0
    num_distances = 0
    for i in range(1, len(sorted_bboxes)):
        if sorted_bboxes[i][1] == sorted_bboxes[i-1][1]:  # Same row
            average_spacing += sorted_bboxes[i][0] - (sorted_bboxes[i-1][0] + sorted_bboxes[i-1][2])  # Gap between boxes
            num_distances += 1

    if num_distances > 0:
        average_spacing /= num_distances
    
    # Define the threshold for a new section based on the average spacing
    threshold = gap_factor * average_spacing if average_spacing > 0 else 0

    # Group bounding boxes into sections
    sections = []
    current_section = []
    for i, bbox in enumerate(sorted_bboxes):
        if i == 0:
            current_section.append(bbox)
        else:
            prev_bbox = sorted_bboxes[i - 1]
            if (bbox[1] == prev_bbox[1] and  # Same row
                bbox[0] - (prev_bbox[0] + prev_bbox[2]) > threshold):  # Gap exceeds threshold
                # Start a new section
                sections.append(current_section)
                current_section = []
            current_section.append(bbox)
    
    if current_section:
        sections.append(current_section)

    return sorted_bboxes

#def draw_sections(img, sections):
    # Loop through each section (list of bounding boxes)
    #for section in sections:
        # Loop through each bounding box in the section
        #for (x, y, w, h) in section:
            # Draw a rectangle around the bounding box (green color, thickness of 2)
            #cv2.rectangle(img, (x, y), (x + w, y + h), (24, 144, 225), 2)
    #return img

#def draw_bounding_boxes(img, bboxes):
    # Loop through each bounding box
    #for (x, y, w, h) in bboxes:
        # Draw a rectangle over the image (green color, thickness of 2)
        #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #return img


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])


path = "C:\\Users\\evans\\Desktop\\Semester Projects\\NetGrowth\\backend\\training_data\\test_images\\test_images\\1f2c589e49a3bcd0.jpg"

train_dataset = ImageFolder(root='./data', train=True, download=True, transform=transform)
test_dataset = ImageFolder(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



#img = cv2.imread(path)
greyScaled = grayScale(img)
ret, im_th = getThreshhold(greyScaled)
sections = createBoundingByLetter(im_th, 2)
bounded = draw_bounding_boxes(img, sections)

cv2.imshow("image window", img)
# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows when any key is pressed
cv2.destroyAllWindows()