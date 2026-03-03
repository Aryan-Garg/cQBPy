function [S, baDemosaicked] = patchMergeBinaryDemosavaria(imbs, flows, param)
%PATCHMERGEBINARYDEMOSAVARIA Merging binary sequence via patchMerge
% demosaicked via Demosavaria
% V2: 
% 210126: use imgScale as demosScale
%Input:
%  imbs: 1D cell array of binary frames
%  flows: coarse flows from alignment algorithm
%  param: struct that contains following fields:
%    patchSizes: array that contains patch sizes for each level.
%                Assumption: (1) size of each level is divisible by the
%                patch size, (2) patchSizes(1) must be an even number
%                Note: only patchSizes(1) is used in merge
%    alignTWSize: temporal window size for alignment
%    alignTWNum: number of temporal windows for alignment
%    mergeTWSize: temporal window size for merging
%    mergeTWNum: number of temporal windows for merging, total number of
%                frames used for merging <= for alignment
%    refFrame: reference frame #
%    imgScale: linear scaling factor for the recovered intensity image
%    wienerC: tuning parameter C for wiener filtering
%    demosaicBinPath: binary executable for demosavaria
%    cfaFile: path for the cfa image file
%    resultDir: directory to save results in
%    debug: whether or not to print debug information
%Output:
%  S: merged sum image
%  baDemosaicked: demosaicked block sum images, shape=[H, W, 3, mergeTWNum]
ts0 = tic;
%% Parameters
if ~iscell(imbs)
    temp = cell(1, size(imbs,3));
    for k = 1:size(imbs,3)
        temp{k} = double(imbs(:,:,k));
    end
    imbs = temp;
end
[H, W, C] = size(imbs{1});
assert(C == 1);
patchSize = param.patchSizes(1);
patchStride = patchSize / 2; % force this to be half patch size
alignTWSize = param.alignTWSize;
alignTWNum = param.alignTWNum;
mergeTWSize = param.mergeTWSize;
mergeTWNum = param.mergeTWNum;

if mergeTWSize * mergeTWNum > alignTWSize * alignTWNum
    error('mergeTWSize * mergeTWNum must be no greater than alignTWSize * alignTNNum!');
end

refFrame = param.refFrame;
if mergeTWSize * mergeTWNum < refFrame
    error('mergeTWSize * mergeTWNum must be no smaller than refFrame!');
end

cenFrame = mod(refFrame - 1, mergeTWSize) + 1; 
refBlock = floor((refFrame - 1) / mergeTWSize) + 1;
param.refImage = refBlock;

% Get the frame number for block i, frame j (i,j starting from 1)
    function idx = frameIdx(i, j)
        idx = (i - 1) * mergeTWSize + j;
    end

% Get the align block and frame subscripts for a give nframe no.
    function [i, j] = alignSub(idx)
        i = floor((idx-1)/alignTWSize)+1;
        j = mod(idx-1, alignTWSize) + 1;
        
    end

mosaickedTempFile = fullfile(param.resultDir, 'temp_mosaicked.tif');
demosaickedTempFile = fullfile(param.resultDir, 'temp_demosaicked.tif');

%% Merge
if alignTWNum == 1
    assert(mergeTWNum == 1 && alignTWSize == mergeTWSize);
    S = zeros(size(imbs{1}));
    for i = 1:alignTWSize
        S = S + imbs{frameIdx(1,i)};
    end
    % demosaic
    temp = mleImage(S, alignTWSize, 1, true);
    if ~isempty(param.imgScale)
        demosScale = 1/min(param.imgScale);
    else
        demosScale = max(temp, [], 'all');
    end
    imwrite(im2uint8(temp/demosScale), mosaickedTempFile);
    command = [param.demosaicBinPath ' ' param.cfaFile ' ' mosaickedTempFile ' ' demosaickedTempFile];
    fprintf(command);
    system(command);
    temp = im2double(imread(demosaickedTempFile)) * demosScale;
    S = mleImageInv(temp(:,:,1:3), alignTWSize, 1);
    fprintf('Merging done. ');
    toc(ts0);
    return
end

% "reference frame" in each align block
alignCenFrame = mod(refFrame - 1, alignTWSize) + 1; 

% Preprocess the flows for interpolation
flowsr = cell(1, alignTWNum+2);
for i = 1:alignTWNum
    flowsr{i+1} = flows{i};
end
flowsr{1} = 2*flowsr{2} - flowsr{3};
flowsr{alignTWNum+2} = 2*flowsr{alignTWNum+1} - flowsr{alignTWNum};

% Function that returns the interpolated flow for a frame
    function iflow = interpFlow(i, j)
        idx = frameIdx(i, j);
        [ai, aj] = alignSub(idx);
        assert(ai > 0 && ai <= alignTWNum && aj > 0 && aj <= alignTWSize);
        if aj < alignCenFrame
            iflow = (alignCenFrame-aj)/alignTWSize*flowsr{ai} + (aj+alignTWSize-alignCenFrame)/alignTWSize*flowsr{ai+1};
        else
            iflow = (alignCenFrame+alignTWSize-aj)/alignTWSize*flowsr{ai+1} + (aj-alignCenFrame)/alignTWSize*flowsr{ai+2};
        end
    end

% First merge each block into a single image
ts = tic;
hs = (H-patchSize)/patchStride+1;
ws = (W-patchSize)/patchStride+1;
xv = repelem((0:ws-1)*patchStride,1,patchSize)+repmat(1:patchSize,1,ws);
yv = repelem((0:hs-1)*patchStride,1,patchSize)+repmat(1:patchSize,1,hs);
[X, Y] = meshgrid(xv, yv);

% First merge each block into a single image and demosaic
baDemosaicked = zeros(H,W,3,mergeTWNum);
fprintf('Demosaicking each block...\n');
for i = 1:mergeTWNum
    blockAverage = zeros(H, W);
    for j = 1:mergeTWSize
        blockAverage = blockAverage + imbs{frameIdx(i,j)};
    end
    % demosaic
    temp = mleImage(blockAverage, mergeTWSize, 1, true);
    if ~isempty(param.imgScale)
        demosScale = 1/min(param.imgScale);
    else
        demosScale = max(temp, [], 'all');
    end
    imwrite(im2uint8(temp/demosScale), mosaickedTempFile);
    command = [param.demosaicBinPath ' ' param.cfaFile ' ' mosaickedTempFile ' ' demosaickedTempFile];
    fprintf(command);
    system(command);
    temp = im2double(imread(demosaickedTempFile)) * demosScale;
    baDemosaicked(:,:,:,i) = mleImageInv(temp(:,:,1:3), mergeTWSize, 1) / mergeTWSize;
    
end
% Then warp and reorganize the images into patches
blockPatches = zeros(hs*patchSize, ws*patchSize, 3, mergeTWNum);
for c = 1:3
    for i = 1:mergeTWNum
        curFlow = interpFlow(i, cenFrame);
        flowwarp = repelem(curFlow, patchSize, patchSize, 1);
        bsWarped = interp2(baDemosaicked(:,:,c,i), X+flowwarp(:,:,1), Y+flowwarp(:,:,2), 'linear');
        bsWarped(~isfinite(bsWarped)) = 0;
        
        blockPatches(:,:,c,i) = bsWarped;
    end
end
toc(ts);

%% Then Wiener Merge
param.H = H; param.W = W;
S = patchMerge(blockPatches, param);
S = S * mergeTWNum * mergeTWSize;
fprintf('Merging done. ');
toc(ts0);
end

