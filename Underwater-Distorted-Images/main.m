%% Adapative_Three_Stages_Reconstruction_for_Underwater_Distorted_Images
% Input：a serie of distorted images or an video of 'avi' format.
% Parameters to be set before running: N(default=80), T(default=3), Its(default=2)
% Edited by Tobez123, 2023.5.22
% Part of codes from Oreifej_2011, James_2019
clc
clear
%% Initialization
global image_height;
global image_width;
addpath(genpath([pwd '\Registration\']));
addpath(genpath([pwd '\inexact_alm_rpca\']));
addpath('YALL1_v1.4');
inputFolder = './datasets/TianAndJames';
USE_GPU = gpuDeviceCount>0; % Set to False if you don't want to use gpu
variable_init();
VideoUtil = VideoUtility();
N=80; %Number of frames we uesd
T = 3;%Threshold for motion sigma after CS
Its = 2; %Number of Iterative times in SBR
%% Input Datasets (format *.bmp or *.avi)
set_flag = 1; % 0 for bmp, 1 for avi
if set_flag ==0
    filepath = uigetdir(inputFolder);
    tag = getFileTag(filepath,set_flag);
    outputFolder = strcat('./results/',num2str(N),'/', tag,'/');
    if ~exist(outputFolder,'dir')
        mkdir(outputFolder);
    end
    frames_list = dir(strcat(filepath,'\*.bmp'));
    frame_num = length(frames_list);
    for i = 1:frame_num
        frame_name = frames_list(i).name;           
        frames(:,:,i) = imread(strcat(filepath,'\',frame_name));
    end
    frames = double(frames(1:image_height, 1:image_width, :)) / 255.0;
elseif set_flag == 1
    filename = uigetfile(strcat(inputFolder, '/*.avi'));
    tag = getFileTag(filename, set_flag);
    outputFolder = strcat('./results/test',num2str(N),'/', tag,'/');
    if ~exist(outputFolder,'dir')
        mkdir(outputFolder);
    end
    filepath = strcat(inputFolder, '/', filename);
    frames = readFrames(filepath);
    frames = frames(1:image_height, 1:image_width, :);
else
    disp('unknown flag')
end

t{1} = clock;
%% Set number o frames
frames = frames(:,:,1:N);
%% Feature Points  Motion detection
[sigma1] = MotionDetection(frames, inf); %calculate the original sigma of video
%% CS Reconstruction
mvfCS = getMotionVectorFieldCS(frames, inf, [8,8,1], [0,0,0], USE_GPU);
frames = im2double(frames);
reconFrames = VideoUtil.WarpVideo(frames, mvfCS, false);
frames = reconFrames;
outfileCS = strcat('\', tag, '-CS');
frames(frames>1) = 1;
frames(frames<0) = 0;
writeOutput1(outputFolder, frames, outfileCS);
t{2} = clock;
%% Feature Points  Motion detection
[sigma2] = MotionDetection(frames, inf);%calculate the sigma of video after CS step
if  sigma2 > 0 && sigma2 <= T

    %% PEOF Reconstruction
    mvfPEOF = getMotionVectorFieldPEOF(frames);
    reconFramesPEOF = VideoUtil.WarpVideo(frames, mvfPEOF, false);
    frames = reconFramesPEOF;
    outfilePEOF = strcat('\', tag, '-PEOF');
    frames(frames>1) = 1;
    frames(frames<0) = 0;
    writeOutput1(outputFolder, frames, outfilePEOF);
    
end
[sigma3] = MotionDetection(frames, inf);%calculate the sigma of video after PEOF step
t{3} = clock;
%% SBR Reconstruction
Means = mean(frames, 3);
imwrite(Means, strcat(outputFolder, tag,'-mean1.bmp'));
for indexLoops = 1:Its
    frame_combine = patch_fuse(frames);
    ImgStatic(:, :, indexLoops) = blind_deconv(frame_combine);
    imwrite(ImgStatic(:, :, indexLoops),strcat(outputFolder,tag,'_blind_deconv',num2str(indexLoops),'.bmp'));
    frames = registration(frames, ImgStatic(:, :, indexLoops), indexLoops);
    Means(:, :, indexLoops+1) = mean(frames, 3);
    imwrite(Means(:, :, indexLoops+1), strcat(outputFolder, tag,'-mean',num2str(indexLoops+1),'.bmp'));
    outfileSBR = strcat('/',tag, '-SBR-',num2str(indexLoops)) ;
    frames(frames>1) = 1;
    frames(frames<0) = 0;
    writeOutput1(outputFolder, frames, outfileSBR);
end
[sigma4] = MotionDetection(frames, inf);%calculate the sigma of video after SBR step
t{4} = clock;
%% RPCA Reconstruction
[a, b, m] = size(frames);
n = a * b;
D = zeros(n,m);
for i = 1:m
    temp = frames(:, :, i);
    D(:, i) = reshape(temp, n, 1);
end
%%inexact_alm_rpca.m
[A_hat, E_hat, iter] = inexact_alm_rpca(D);

if ~exist(strcat(outputFolder,'\A_hat_', tag, '\'), 'dir')
    mkdir(strcat(outputFolder,'\A_hat_', tag, '\'));
end
if ~exist(strcat(outputFolder,'\E_hat_', tag, '\'), 'dir')
    mkdir(strcat(outputFolder,'\E_hat_', tag, '\'));
end

imA = zeros(size(frames));
imE = zeros(size(frames));
for i=1:m
    imA(:, :, i) = reshape(A_hat(:, i), [size(frames, 1), size(frames, 2)]);
    imwrite(imA(:, :, i), strcat(outputFolder, 'A_hat_', tag,'\', sprintf('Image_%.3d.bmp', i)));
    
    imE(:, :, i) = reshape(E_hat(:, i), [size(frames, 1), size(frames, 2)]);
    imwrite(imE(:, :, i), strcat(outputFolder, 'E_hat_', tag,'\', sprintf('Image_%.3d.bmp', i)));
end
imA(imA>1) = 1;
imA(imA<0) = 0;
imE(imE>1) = 1;
imE(imE<0) = 0;
outfileimA = strcat('A_hat_', tag);
outfileimE = strcat('E_hat_', tag);
writeOutput1(outputFolder, imA, outfileimA);
writeOutput1(outputFolder, imE, outfileimE);
t{5} = clock;
% Save images and video

output1 = mean(imA, 3);
output2 = patch_fuse(imA);
outputName1 = strcat(outputFolder, 'result', tag,'-mean.bmp');
imwrite(output1, outputName1);
outputName2 = strcat(outputFolder, 'result', tag, '-patchfusion.bmp');
imwrite(output2, outputName2);

%% Save time
time = zeros(size(t, 2), 1);
for i = 1:size(t, 2)-1
    time(i) = etime(t{i+1}, t{i});
end
time(i+1) = etime(t{i+1}, t{1});
save(strcat(outputFolder,'time.mat')','time');
%% Save sigma
sigma = [sigma1 sigma2 sigma3 sigma4];
save(strcat(outputFolder,'sigma.mat')','sigma');


%%  functions used in algorithm
function []=writeOutput1(outFol, reconFrames, name)
    file = strcat(outFol,name,'.avi');
    nrFrames = size(reconFrames,3);
    v = VideoWriter(file,'Motion JPEG AVI');
    v.Quality = 100;
    v.FrameRate = 25;
    open(v);
    for i = 1:nrFrames
        outFrame = squeeze(reconFrames(:,:,i));
        writeVideo(v,outFrame);
    end
end

function tag =  getFileTag(file,k)
            if k==0
                a = strsplit(file,'\');
                tag = convertCharsToStrings(a{end});
            elseif k==1
                a = strsplit(file, '.');
                tag = convertCharsToStrings(a{end-1});
            else
                disp('unknown flag');
            end
end
        
 function frames = readFrames(filePath)
        v = VideoReader(filePath);
            
        frames = read(v);
        frames = squeeze(mean(frames,3))/255;
            
 end
        


