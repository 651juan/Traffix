% VIDEO INFO - 11 FPS

function multiObjectTracking()
%Setup vlFeat
warning off;
%JUAN PC
%run ('vlfeat\toolbox\mex\mexw64\vl_version.mexw64')
%run('C:\MATLAB\ImageCateogrisation\vlfeat\toolbox\vl_setup');
%ADRIAN PC
run('C:\Users\Admin\Documents\MATLAB\Extra\vlfeat\toolbox\vl_setup');

%Load Masks
ATTARD_MASK = single(rgb2gray(imread('attardmask.jpg')))./255;

%Load PreTrained Data
load('positiveHistograms.mat');
load('negativeHistograms.mat');
load('cTotalDescriptors.mat');

%Setup Detector to use
detector = 'SIFT'; %SIFT/SURF/ESURF


%Setup Training Data
trainingData = [positiveHistograms(1:end,:);negativeHistograms(1:end,:)];
trainingLabels = [ones(size(positiveHistograms,1),1);zeros(size(negativeHistograms,1),1)];
%Train KNN Model
KNNTrainedModel = fitcknn(trainingData, trainingLabels);

%Setup Video Paths
ATTARD_VIDS_PATH = 'C:\Users\Admin\Documents\GIT\skylinewebcamcrawler\videos\Attard, Mdina Road from Citroen Showroom\';

if ~exist('output.json')
    fileID = fopen('output.json','w');
    fprintf(fileID,'{\n');
    fclose(fileID);
end

while(true)
    direc = dir(strcat(ATTARD_VIDS_PATH,'*.mp4'));
    currFile = strcat(ATTARD_VIDS_PATH,direc(1).name)
    tic;
    [carSpeed,carCount] = trackVideo(currFile,ATTARD_MASK,KNNTrainedModel,cTotalDescriptors, detector);
    elapsedTime = toc
    
    if(carSpeed <= 35) && (carCount >=15)
        isTrafficResult = 'true'
    else
        isTrafficResult = 'false'
    end
    
    fileID = fopen('output.json','a');
    toOutput = strcat('"', char(datetime('now')),'":[');
    toOutput = strcat(toOutput,'{','"LOCATION": "ATTARD", "TRAFFIC":"',isTrafficResult, '","SPEED":"', num2str(carSpeed), '","COUNT":"', num2str(carCount),'"}');
    toOutput = strcat(toOutput,'],\n');
    fprintf(fileID,toOutput);
    fclose(fileID);
    
    currentFileContents = fileread('output.json');
    currentFileContents = currentFileContents(1:size(currentFileContents,2)-2); %Remove las newline and comma
    file2 = fopen('resultsToUpload.json','w');
    toOutput = strcat(currentFileContents,'\n}\n');
    fprintf(file2,toOutput);
    fclose(fileID);
    
    %Setup FTP
    ftpObject = ftp('ftp.nnjconstruction.com:21','lifex@nnjconstruction.com','Lifex..2016');
    mput(ftpObject,'resultsToUpload.json');
    close(ftpObject);
    
    videosToDelete = ceil((elapsedTime ./ 60) ./ 2);
    
    for i = 1:videosToDelete
        delete(strcat(ATTARD_VIDS_PATH,direc(i).name));
    end
end
end

function [carSpeed,carCount] = trackVideo(videoPath,videoMask,KNNTrainedModel,cTotalDescriptors, detector)

videoInfo = VideoReader(videoPath);
videoDur = videoInfo.Duration;
videoFPS = videoInfo.FrameRate;
lastFrame = round(videoDur*videoFPS);

%Setup System
obj = setupSystemObjects(videoPath);

opticFlow = opticalFlowLK('NoiseThreshold',0.009);

% Detect moving objects, and track them across video frames.
frameCount = 0;

mag = [];
count = [];

tenFG = 0;
elevenFG = 0;
%figure;
while ~isDone(obj.reader)
    if frameCount == tenFG && frameCount ~= elevenFG
        tenFG = tenFG + (videoFPS-1);
        %Read next frame and apply mask
        frame = readFrame(obj, videoMask, true);
        %Transform image into blob image
        blobImg = createBlobImage(frame, obj, true);
        flow = estimateFlow(opticFlow,blobImg);
    elseif frameCount == tenFG && frameCount == elevenFG
        tenFG = tenFG + (videoFPS-1);
    end
    
    %Every 11 Frames ie every second calculate speed and car count
    if frameCount == elevenFG
        elevenFG = elevenFG + videoFPS;
        %Read next frame and apply mask
        frame = readFrame(obj, videoMask, true);
        %Transform image into blob image
        blobImg = createBlobImage(frame, obj, true);
        %detect velocity
        flow = estimateFlow(opticFlow,blobImg);
        Vx = nansum(nansum(flow.Vx));
        Vy = nansum(nansum(flow.Vy));
        mag(end+1) = sqrt(Vx^2 + Vy^2);
        count(end+1) = getNumberOfCars(KNNTrainedModel,frame,9, cTotalDescriptors, detector);
        %fprintf( 'Car Speed: %d\n', mag(end));
        %fprintf( 'Car Count: %d\n\n', count(end));
        %imshow(frame);
    elseif frameCount ~= tenFG
        %Read next frame and apply mask
        frame = readFrame(obj, videoMask, false);
        %Transform image into blob image
        blobImg = createBlobImage(frame, obj, false);
    end
    
    frameCount = frameCount +1;
    
    %At the end of the 2 mins (11 frames per second *120 seconds) = 1320 frames
    %if frameCount == 1320
    if frameCount == lastFrame
        magMedian = median(mag);
        countMedian = round(median(count));
        
        if (countMedian ~= 0)
            magMedian = (magMedian / countMedian);
        else
            magMedian = 0;
        end
        
        carSpeed = magMedian;
        carCount = countMedian;
        return
    end
end

end

function obj = setupSystemObjects(vidPath)
% Create a video file reader.
obj.reader = vision.VideoFileReader(vidPath);

% Create System objects for foreground detection and blob analysis

% The foreground detector is used to segment moving objects from
% the background. It outputs a binary mask, where the pixel value
% of 1 corresponds to the foreground and the value of 0 corresponds
% to the background.

obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
end

%reads a frame and applies a mask to remove extra detail
function frame = readFrame(obj, mask, check)
frame = obj.reader.step();
if(check)
    frame(:,:,1) = frame(:, :, 1).*mask;
    frame(:,:,2) = frame(:, :, 2).*mask;
    frame(:,:,3) = frame(:, :, 3).*mask;
end
end

%returns a blob image (black/white)
function mask = createBlobImage(frame,obj, check)
% Detect foreground.
mask = obj.detector.step(frame);

if(check)
    % Apply morphological operations to remove noise and fill in holes.
    mask = imopen(mask, strel('rectangle', [3,3]));
    mask = imclose(mask, strel('rectangle', [15, 15]));
    mask = imfill(mask, 'holes');
end
end

%Returns the number of cars in an image using a pretrained knn model
function result = getNumberOfCars(trainingModel, frame, parts, cTotalDescriptors, detector)
if ndims(frame) == 3
    frame = single(rgb2gray(frame));
else
    frame = single(frame);
end

step1 = floor(size(frame,1)./parts);
step2 = floor(size(frame,2)./parts);

imggrid = [];
for i = 1:step1:size(frame,1)-step1
    for j = 1:step2:size(frame,2)-step2
        tmp = frame(i:i+step1-1,j:j+step2-1);
        imggrid = [imggrid;getSingleBagOfWords(tmp, cTotalDescriptors, detector)];
    end
end
pred = predict(trainingModel, imggrid);
result = sum(pred);
end

%Returns a bag of words histogram of an image using the given bins
function bag = getSingleBagOfWords(im, bins, detector)
[~,desc] = runDetector(im,detector);
bag = zeros(1,size(bins,2));

for j = 1:size(desc,2)
    dist = [];
    for k = 1:size(bins,2)
        dist(k) = norm(double(desc(:,j))-bins(:,k));
    end
    [~,bin] =  min(dist);
    bag(bin) = bag(bin)+1;
end
end

function [f,d] = runDetector(image, detector)
if(strcmp(detector,'SIFT'))
    if ndims(image) == 3
        image = single(rgb2gray(image));
    else
        image = single(image);
    end
    
    [f,d] = vl_sift(image);
elseif ((strcmp(detector,'SURF')) || (strcmp(detector,'ESURF')))
    if ndims(image) == 3
        image = rgb2gray(image); %Do not convert to single when using SURF
    end
    points = detectSURFFeatures(image);
    if(strcmp(detector,'SURF'))
        [d, f] = extractFeatures(image, points);
    elseif (strcmp(detector,'ESURF'))
        [d, f] = extractFeatures(image, points, 'SURFSize',128);
    end
    
    d = d';
end


d = normc(single(d));
d(d > 0.2) = 0.2;
d = normc(d);
end
