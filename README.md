Vehicle Detection And Counting using OpenCV
This project uses OpenCV to detect and count vehicles in a video. The project uses a background subtraction algorithm to identify moving objects in the video. The moving objects are then classified as vehicles using a simple shape analysis algorithm. The vehicles are then counted and the results are displayed on the video.

The project consists of the following files:

main.py - The main Python script that runs the project.
haarcascade_car.xml - A pre-trained Haar cascade classifier for detecting cars.
video.mp4 - The video file that the project will detect and count vehicles in.
To run the project, open a terminal window and navigate to the directory where the project files are located. Then, type the following command:

python main.py
The project will start running and the results will be displayed on the video.

How does it work?
The project works in the following steps:

The video file is read into memory.
A background subtraction algorithm is applied to the video to identify moving objects.
The moving objects are then classified as vehicles using a simple shape analysis algorithm.
The vehicles are then counted and the results are displayed on the video.
Background subtraction algorithm
The background subtraction algorithm used in this project is the Gaussian Mixture Model (GMM) algorithm. The GMM algorithm works by modeling the background as a mixture of Gaussian distributions. The moving objects are then identified as the pixels that do not belong to the background model.

Shape analysis algorithm
The shape analysis algorithm used in this project is a simple algorithm that checks the width and height of the moving objects. The objects that are wider than a certain threshold and taller than a certain threshold are classified as vehicles.

Future work
There are a number of ways that this project could be improved in the future. One way would be to use a more sophisticated background subtraction algorithm. Another way would be to use a more sophisticated shape analysis algorithm. Additionally, the project could be extended to detect and count other types of objects, such as pedestrians and bicycles.