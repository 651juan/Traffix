function trainSiftCarDetector()
run('C:\Users\Admin\Documents\MATLAB\Extra\vlfeat\toolbox\vl_setup');

positiveDataDirec = 'Datasets\POSITIVE\POS_SET_3\';
positiveDataExtension = 'jpg';
negativeDataDirec = 'Datasets\NEGATIVE\';
negativeDataExtension = 'jpg';
kMeansLimit = 100;
kMeansMaxIter = 1000;
distanceMes = 'cosine';
detector = 'SIFT'; %SIFT/SURF/ESURF

%%POSITIVE SET
if ~exist('ucPtrain.mat') %|| 1 == 1
    fprintf('Getting Positive Descriptors\n');
    ucPtrain = getUnclusteredDescriptors(positiveDataDirec,positiveDataExtension,detector);
    save ucPtrain ucPtrain;
else
    load('ucPtrain.mat');
end

%NEGATIVE SET

if ~exist('ucNtrain.mat') %|| 1 == 1
    fprintf('Getting Negative Descriptors\n');
    ucNtrain = getUnclusteredDescriptors(negativeDataDirec,negativeDataExtension, detector);
    save ucNtrain ucNtrain;
else
    load('ucNtrain.mat');
end

%Merge Descriptors and Cluster
totalUCDescriptors = [ucPtrain ucNtrain];

if ~exist('cTotalDescriptors.mat') %|| 1 == 1
    fprintf('Clustering Descriptors\n');
    cTotalDescriptors = getClusteredDescriptors(totalUCDescriptors,kMeansLimit,distanceMes,kMeansMaxIter);
    save cTotalDescriptors cTotalDescriptors;
else
    load('cTotalDescriptors.mat');
end

%Describe photos using bag of visual words
if ~exist('positiveHistograms.mat') %|| 1 == 1
    fprintf('Creating Positive Histograms\n');
    positiveHistograms = getBagOfWords(positiveDataDirec,positiveDataExtension,cTotalDescriptors, detector);
    save positiveHistograms positiveHistograms;
else
    load('positiveHistograms.mat');
end

if ~exist('negativeHistograms.mat') %|| 1 == 1
    fprintf('Creating Negative Histograms\n');
    negativeHistograms = getBagOfWords(negativeDataDirec,negativeDataExtension,cTotalDescriptors, detector);
    save negativeHistograms negativeHistograms;
else
    load('negativeHistograms.mat');
end

trainingData = [positiveHistograms(1:500,:);negativeHistograms(1:1000,:)];
trainingLabels = [ones(500,1);zeros(1000,1)];
testingData = [positiveHistograms(501:end,:);negativeHistograms(1001:end,:)];
testLabels = [ones(size(positiveHistograms(501:end,:),1),1);zeros(size(negativeHistograms(1001:end,:),1),1)];

prediction = knnclassify(testingData,trainingData,trainingLabels);
accuracy = sum(testLabels == prediction)/size(testingData,1)

testImg = imread('test7.jpg');

parts = 9;
step1 = floor(size(testImg,1)./parts);
step2 = floor(size(testImg,2)./parts);

imggrid = [];
for i = 1:step1:size(testImg,1)-step1
    for j = 1:step2:size(testImg,2)-step2
        tmp = testImg(i:i+step1-1,j:j+step2-1);
%         figure;
%         imagesc(tmp);
        imggrid = [imggrid;getSingleBagOfWords(tmp, cTotalDescriptors, detector)];
    end
end

pred = knnclassify(imggrid,trainingData,trainingLabels)
result = sum(pred)
end

function bag = getSingleBagOfWords(im, bins, detector)
[~,desc] = runDetector(im, detector);
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

function Histograms = getBagOfWords(direc, extension, bins, detector)
d = dir(strcat(direc,'*.',extension));
Histograms = [];
for i = 1:numel(d)
    im = imread(strcat(direc,d(i).name));
    
    [~,desc] = runDetector(im, detector);
    bag = zeros(1,size(bins,2));
    
    for j = 1:size(desc,2)
        dist = [];
        for k = 1:size(bins,2)
            dist(k) = norm(double(desc(:,j))-bins(:,k));
        end
        [~,bin] =  min(dist);
        bag(bin) = bag(bin)+1;
    end
    Histograms = [Histograms;bag];
end
end

function clusteredResults = getClusteredDescriptors(unclusteredDescriptors, clusterLimit, distance, maxIterations)
[minidx,clusteredResults,~] = DunnsIndex(double(unclusteredDescriptors'), clusterLimit, distance, maxIterations);
minidx
end

function unclusteredDesc = getUnclusteredDescriptors(direc,extension, detector)
d = dir(strcat(direc,'*.',extension));
unclusteredDesc = [];

for i = 1:numel(d)
    im = imread(strcat(direc,d(i).name));
    
    [~,desc] = runDetector(im, detector);
    unclusteredDesc = [unclusteredDesc desc];
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