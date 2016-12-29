# Traffix

This Library is written in MATLAB and enables users to determine if there is traffic in a particular video.

This is done by determining the velocity and count the number of cars in the video and then calculating the average speed of the video sequence. Since videos capture devices can be of difference distances from the road, the user has to identify the limit of the average speed to be considered as traffic.

Also, the library uses masks on video to cut off any unwanted areas in the video such as pavements or lanes which you do not want to consider. 

The library requires the VLFeat Library to perform SIFT(http://www.vlfeat.org/) and a dataset of cars to train the detector.

The Code can upload the data to a server and viewed on smartphones using the android application below.

# USAGE

1. Download a car dataset and place it in the TrainDetector folder with a 'NEGATIVE' and 'POSITIVE' sub folder structure.
2. Run the file 'trainSiftCarDetector.m' which will produce a 'negativeHistogram.mat', 'positiveHistogram.mat' and 'cTotalDescriptors.mat'
3. Copy these .mat files in the main folder.
4. Download the vlfeat Library and run its setup.
5. Edit the multiObjectTracking.m script to change the video file path and the mask path.
6. Results will be outputted to console.

# Android Application 
For the android application follow the link (https://github.com/651juan/Traffixui).
