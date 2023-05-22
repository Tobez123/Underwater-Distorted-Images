function [sigma] = MotionDetection(frames, nrPts)
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
nrFrames = size(frames, 3);
pointTracker = vision.PointTracker('BlockSize', [21, 21]);
firstFrame = squeeze(frames(:, :, 1));

points = detectInterestPoints(firstFrame);
points = removeBorderPts(points, [size(frames, 1), size(frames, 2)], 24);
nrPts = min(nrPts, size(points, 1));
points = points(1:nrPts, :);
initialize(pointTracker, points, firstFrame);

tracks = zeros(nrPts, nrFrames, 2);
scores = zeros(nrPts, nrFrames);
validity = true;
for i = 1:nrFrames
    frame = squeeze(frames(:,:,i));
    [points, point_validity, scores(:, i)] = pointTracker(frame);
    tracks(:,i,:) = points(:, end:-1:1);
    validity = validity&point_validity;
end
tracks = tracks(validity, :, :);
tracks = filterCotShifts(tracks);

cots = mean(tracks, 2);
tracks = tracks - cots;
%cots = squeeze(cots);
dist1 = zeros(size(tracks, 1), size(tracks, 2));
dist2 = zeros(size(tracks, 1), 1);
for i = 1:size(tracks, 1)
    for j = 1:size(tracks,2)
        dist1(i, j) = (tracks(i,j,1))^2 + (tracks(i,j,2))^2;
    end
    dist2(i) = sum(dist1(i,:));
end
sigma= sqrt(sum(dist2)/(size(tracks, 1)*size(tracks, 2) - 1));

end

%% Functions used in algorithm

function points = detectInterestPoints(frame)

surf = detectSURFFeatures(frame);
brisk = detectBRISKFeatures(frame);
fast = detectFASTFeatures(frame);
harris = detectHarrisFeatures(frame);
%kaze = detectKAZEFeatures(frame);
%orb = detectORBFeatures(frame);
points = joinPts({surf, brisk, fast, harris}, size(frame));
end

 
function points = joinPts(pts, shape)
points = [];

for i = 1:length(pts)
    sPts = pts{i};
    points = vertcat(points,sPts.Location);
end
points = removeDuplicates(points, shape);
 end

 
function points =  removeBorderPts(pts, shape, clr)
validR = ((pts(:,2)>clr) & (pts(:,2)<(shape(1)-clr)));
validC = ((pts(:,1)>clr) & (pts(:,1)<(shape(2)-clr)));

validPts = validR & validC;
points = pts(validPts,:);
end


function tracks = filterCotShifts(tracks)
mid = floor(size(tracks,2)/2);
cotF = mean(tracks(:,1:mid,:),2);
cotS = mean(tracks(:,mid+1:2*mid,:),2);

dists = squeeze(sqrt(sum((cotF-cotS).^2,3)));

validTracks = dists<3;

tracks = tracks(validTracks,:,:);

end


function points = removeDuplicates(pts, shape)
pts = round(pts);
idxs = sub2ind(shape,pts(:,2), pts(:,1));
[~, idxs] = unique(idxs);
points = pts(idxs,:);


end