from torch import rand
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import random

from backend.customDataSet import CustomImageDataset


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])

def createRandomKernel(channels, size):
    kernel = []
    for h in range(channels):
        kernel.add([])
        for i in range(size):
            kernel[h].add([])
            for j in range(size):
                kernel[h][i].add(random.uniform(0, 1))
    return kernel

    #create Kernel of sizexsize with random values
    #rand(size)


def addPaddingToMatrix(matrix, paddingSize):
    length = len(matrix[0])-1

    #insert new row at 0
    matrix.insert(0, [])

    #for each column, add a 0
    for i in range(length):
        matrix[0].add(0)

    #add new row at bottom
    matrix.add([])

    #for each column add a 0
    for i in range(length):
        matrix[len(matrix)-1].add(0)

    #for all the middle rows, add a 0 at the beginning and end
    for i in range(1, len(matrix)-1):
        matrix.insert(0,0)
        matrix.add(0)

    return matrix

def convertImageToTensor(img): #Tensor is essentially an array of Matrices. since using greyscale, it's only 1 matrix stored in the array
    
    #convert an image to a tensor, return resulting tensor
    print("foo")

def elementWiseMultiplication(inputMatrix, targetRow, targetColumn, kernelMatrix, outputLayer, outputRow, outputColumn):
    #elementWise multiply around the target in the matrix, using the kernel, insert into the outputLayer at outputRow, outputColumn
    print("foo")

def buildOutputLayer(image, kernel, padding, stride, channels):
    output = []
    rowsOfMatrix = len(image[h])
    columnsOfMatrix = len(image[h][0])
    rowsOfKernel = len(kernel[h])
    columnsOfKernel = len(kernel[h][0])
    outputRows = (rowsOfMatrix -  rowsOfKernel + (2*padding))/stride
    outputColumns = (columnsOfMatrix - columnsOfKernel + (2*padding))/stride
    
    for h in range(channels):
        currentRow = -1
        currentColumn = -1

        totalOutputLayer= []
        outputLayer = [] #Tensor that is outputRows X outputColumns
        for i in range(padding, outputRows, stride):
            currentRow += 1
            outputLayer.add([])
            for j in range(padding, outputColumns, stride):
                currentColumn+=1
                outputLayer[i].add(0)
                #loop through each item in our tensor, call elementWiseMultiplication for each item, setting outputLayer
                outputLayer = elementWiseMultiplication(image[h], i, j, kernel[h], outputLayer, currentRow, currentColumn)
                if len(totalOutputLayer = 0):
                    totalOutputLayer = outputLayer
                else:
                    for row in outputLayer:
                        for column in row:
                            totalOutputLayer[row][column] = totalOutputLayer[row][column] + outputLayer[row][column]
    #output.add(outputLayer)
    return output

    
def buildKernels(channels, kernels, kernelCount, kernelSize):
    for i in range(kernelCount):
        kernels.add(createRandomKernel(channels, kernelSize))
    return kernels

def ReLu(matrix):
    rowsOfTensor = len(matrix)
    columnsOfTensor = len(matrix[0])
    for i in range(0, rowsOfTensor):
        for j in range(0, columnsOfTensor):
            matrix[i][j] = max(0, matrix[i][j])
    return matrix

def MaxPoolTensor(matrix, size):
    rowsOfTensor = len(matrix)
    columnsOfTensor = len(matrix[0])
    outputTensor = "" #Tensor of size Ceil((rowsOfTensor/size)) x Ciel((columnsOfTensor/size)))
    currentRow = -1
    currentColumn = -1
    for i in range(0, rowsOfTensor, size):
        currentRow+=1
        for j in range(0, columnsOfTensor, size):
            currentColumn+=1
            outputTensor[currentRow][currentColumn] = max(0, matrix[i][j])
    return matrix

def main():
    train_dataset = CustomImageDataset(img_dir='./training_data/train', transform=transform)
    test_dataset = CustomImageDataset(img_dir='./training_data/test', transform=transform)
    padding = 1
    stride = 1
    pooling = 2
    channels = 1
    num_epochs = 10
    convolutional_layers = 2
    batch_size = 64
    kernel_count = 8
    kernels = []
    kernels = buildKernels(channels, kernels, kernel_count, 3)



    tensor = []

        #startLoop
    for epoch in num_epochs:
        for batch in batch_size:
            tensor = tensor.add(convertImageToTensor(train_dataset.__getitem__(random(0, len(train_dataset.image_filenames)-1))))

        for channel in tensor:
            for matrix in channel:
                matrix = addPaddingToMatrix(matrix, padding)




        featureMaps=[]
        for layer in convolutional_layers:
            for kernel in kernels:
                for image in tensor:
                    featureMaps.add(buildOutputLayer(image, kernel, padding, stride, channels))
            kernels = buildKernels(channels, kernels, kernel_count, 3)
            kernel_count = kernel_count*2
            


        for featureMap in featureMaps:
            featureMap = MaxPoolTensor(featureMap, pooling)
            featureMap = ReLu(featureMap)
    
        tensor = featureMaps


    print("foo")

#path = "C:\\Users\\evans\\Desktop\\Semester Projects\\NetGrowth\\backend\\training_data\\test_images\\test_images\\1f2c589e49a3bcd0.jpg"

#batch_size = 64
#train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



#img = cv2.imread(path)
#greyScaled = grayScale(img)
#ret, im_th = getThreshhold(greyScaled)
#sections = createBoundingByLetter(im_th, 2)
#bounded = draw_bounding_boxes(img, sections)

#cv2.imshow("image window", img)
# Wait indefinitely until a key is pressed
#cv2.waitKey(0)

# Close all OpenCV windows when any key is pressed
#cv2.destroyAllWindows()