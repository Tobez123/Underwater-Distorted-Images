%%  functions used in algorithm
function []=writeOutput1(outFol, reconFrames, name)
    file = strcat(outFol,name,'.avi');
    nrFrames = size(reconFrames,3);
    v = VideoWriter(file,'Motion JPEG AVI');
    v.Quality = 100;
    v.FrameRate = 50;
    open(v);
    for i = 1:nrFrames
        outFrame = squeeze(reconFrames(:,:,i));
        writeVideo(v,outFrame);
    end
end