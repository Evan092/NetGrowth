Final
[Evan Snowgold]
[esnowgol@emich.edu]
December 10, 2024

Video:
https://drive.google.com/file/d/14iVHsARd67QYdWkTXBe2DBxJt5zrY7Ji/view?usp=sharing

Usage[1]
1. Using command prompt, cd to NetGrowth folder
2. Load Conda Enviroment using environment.yml (conda env create -f environment.yml)
3. In the command prompt, cd into the backend folder.
4. Run main.py.
5. Insert a path (with or without quotation marks).
6. This should download YOLOv5 (or load it from cache).
7. Various files will be output, and a path will be printed:
    • Images with predicted bounding boxes on them.
    • Cropped resized texts results in a folder (image file named using prediction).
    • output.txt, containing all the predicted text in order based on the bounding box position.
8. Outputs should be in NetGrowth/backend/training data/verify/epoch 0.


How to train boundingBoxCNN?
The BoundingBox CNN was trained using varying methods across 213 epochs. While I ended up at my
goal, I altered the loss function along the way (removing SmoothL1Loss, altering weights, etc.) Unsure
of how training from scratch again using this file would go. I used a 3090 for training, so depending on
hardware, batch size may need decreasing. Already set to continue from the latest checkpoint, just need
to run and watch after adding training data.


How to train TextCRNN?
Works with CustomDataSet2 and CustomDataSet3. CustomDataSet2 uses the full image, crops, and
resizes around the text (slow). Encounters errors sometimes due to random rotations sometimes causing
the bounding box to be off the picture (doesn’t stop training, fails gracefully). CustomDataSet3 Uses
pre-processed images to speed up training. Already set to continue from the latest checkpoint. Just run
and watch after adding training data.
Possible Questions


What’s with ”frontend - abandoned”?
Originally, the project was intended to include a front-end phone app. However, neural networks became
my primary focus and took more time than anticipated. As a result, the front-end was deemed unneces-
sary and ultimately abandoned. Despite this, the work I completed is still included in the codebase but
can likely be ignored.

I want to train, but where is your training data?
My dataset is extensive, with additional data generated during preprocessing. Due to its size, the dataset
is not included directly in the project files. We met during office hours, it’s unlikely you want to download
21GB of training data, but I didn’t get home in time to stop the upload, so it is available.[2]
References
[1] Evan Snowgold. Netgrowth. https://github.com/esnowgol/NetGrowth, 2024.
[2] TextOCR. Textocr. https://drive.google.com/file/d/1m2J6fUZwPtz4OiVaRcPQXg5CzL-utsDq/
view?usp=sharing, 2024.

Abandoned front end screenshots:

![Screenshot_20240930_222052_Expo Go](https://github.com/user-attachments/assets/3a2c4476-4796-48c3-97cb-64fbb4dff102)
![Screenshot_20240930_222058_Expo Go](https://github.com/user-attachments/assets/c62b13b8-4feb-47e3-a3f3-bebdb2958618)
![Screenshot_20240930_222115_Expo Go](https://github.com/user-attachments/assets/01b2e419-ecfc-4c84-a83b-2ca038a2f659)
