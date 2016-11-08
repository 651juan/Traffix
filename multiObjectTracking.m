function multiObjectTracking()
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();

opticFlow = opticalFlowLK('NoiseThreshold',0.009);

% Detect moving objects, and track them across video frames.
frameCount = 0;
load('positiveHistograms.mat');
load('negativeHistograms.mat');
load('cTotalDescriptors.mat');
trainingData = [positiveHistograms(1:end,:);negativeHistograms(1:end,:)];
trainingLabels = [ones(size(positiveHistograms,1),1);zeros(size(negativeHistograms,1),1)];

modelTraining = fitcknn(trainingData, trainingLabels);
mag = [];
count = [];
    while ~isDone(obj.reader)
        frame = readFrame(obj);
        mask = detectObjects(frame, obj);
        %detect velocity
        flow = estimateFlow(opticFlow,mask);
        if mod(frameCount, 11) == 0
            Vx = nansum(nansum(flow.Vx));
            Vy = nansum(nansum(flow.Vy));
            mag(end+1) = sqrt(Vx^2 + Vy^2);
            count(end+1) = getNumberOfCars(modelTraining,frame,9, cTotalDescriptors);
            disp(mag);
            disp(count);
        end
        
        if frameCount == 1320 
            magMedian = median(mag);
            countMedian = median(count);
            disp('ANSWER: ');
            if (countMedian == 0)
                disp(magMedian / countMedian);
            else 
                disp(0);
            end
            return
        end
        frameCount = frameCount +1;
    end
end

function obj = setupSystemObjects()
% Initialize Video I/O
% Create objects for reading a video from a file, drawing the tracked
% objects in each frame, and playing the video.

% Create a video file reader.
obj.reader = vision.VideoFileReader('test1.mp4');

% Create two video players, one to display the video,
% and one to display the foreground mask.
obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);
obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);

% Create System objects for foreground detection and blob analysis

% The foreground detector is used to segment moving objects from
% the background. It outputs a binary mask, where the pixel value
% of 1 corresponds to the foreground and the value of 0 corresponds
% to the background.

obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
    'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);

%Setup vlFeat
run ('vlfeat\toolbox\mex\mexw64\vl_version.mexw64')
run('C:\MATLAB\ImageCateogrisation\vlfeat\toolbox\vl_setup');
end

function frame = readFrame(obj)
amask = single(rgb2gray(imread('attardmask.jpg')))./255;
frame = obj.reader.step();
frame(:,:,1) = frame(:, :, 1).*amask;
frame(:,:,2) = frame(:, :, 2).*amask;
frame(:,:,3) = frame(:, :, 3).*amask;
end

function mask = detectObjects(frame,obj)

% Detect foreground.
mask = obj.detector.step(frame);

% Apply morphological operations to remove noise and fill in holes.
mask = imopen(mask, strel('rectangle', [3,3]));
mask = imclose(mask, strel('rectangle', [15, 15]));
mask = imfill(mask, 'holes');

end

function result = getNumberOfCars(trainingModel, frame, parts, cTotalDescriptors)


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
        imggrid = [imggrid;getSingleBagOfWords(tmp, cTotalDescriptors)];
    end
end
pred = predict(trainingModel, imggrid);
result = sum(pred);
end

function bag = getSingleBagOfWords(im, bins)
if ndims(im) == 3
    im = single(rgb2gray(im));
else
    im = single(im);
end

[~,desc] = runSift(im);
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

function [f,d] = runSift(image)
[f,d] = vl_sift(image);
d = normc(single(d));
d(d > 0.2) = 0.2;
d = normc(d);
end
