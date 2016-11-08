function [minidx,centroids,labels] = DunnsIndex(data, limit, distance, maxIterations)

validity = zeros(1, limit);

kmresults = cell(limit,2);
for k = 2:limit
   fprintf('Working KMeans for k = %d of %d\n',k,limit);
   [kmresults{k,1},kmresults{k,2}] = kmeans(data,k,'distance',distance,'MaxIter', maxIterations);  
   validity(k) = getDunnsIndex(data,kmresults{k,1});
end

validity = validity(2:numel(validity));
[~,minidx] = min(validity);

minidx = minidx + 1;
centroids = kmresults{minidx,2};
centroids = centroids';
labels = kmresults{minidx,1};
labels = labels';
end

function validity = getDunnsIndex(data,labels)

%intra variance
intra = 0;
for i = 1:max(labels)
    datapoints = data(labels == i,:);
    centroid = mean(datapoints);
    
    for j = 1:size(datapoints,1)
        intra = intra + norm(datapoints(j,:) - centroid); %norm computes magnitude of this distance which is the euclidian distance
    end
end

intra = intra ./ size(data,1);

%inter variance
interlist = [];
for i = 1:max(labels)-1
    datapoints = data(labels == i,:);
    centroid_i = mean(datapoints);
    
    for j = i+1:max(labels)
        datapoints = data(labels == j,:);
        centroid_j = mean(datapoints);
        interlist(end + 1) = norm(centroid_i - centroid_j);       
    end
end
inter = min(interlist);

validity = intra/inter;
end
