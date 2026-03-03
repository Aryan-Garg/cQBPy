function [ima, S] = naiveReconsDemosavaria(imbs, param)
%NAIVERECONSDEMOSAVARIA Naive reconstruction by simple summing the binary images and
%compute MLE
%V2: 
%210116: demoScale = imgScale
%220227: demoScale = max(im, 'all') if param.imgAutoScale
%220512: make use of cropSize
%Input:
%  imbs: 1D cell array of binary frames
%  param: struct that contains following fields:
%    mergeTWSize: window size for temporal reconstruction
%    mergeTWNum: number of temporal windows (determines total number of
%                frames being used), > 2
%    refFrame: reference frame #
%    imgScale: linear scaling factor for intensity image
%    debug: whether or not to print debug information
%Output:
%  ima: reconstructed image
%  S: sum image

%% Parameters

if ~iscell(imbs)
    temp = cell(1, size(imbs,3));
    for k = 1:size(imbs,3)
        temp{k} = double(imbs(:,:,k));
    end
    imbs = temp;
end

N = numel(imbs);
% fprintf('%d frames\n', N);
[H, W, C] = size(imbs{1});
assert(C==1);
twSize = param.mergeTWSize;
twNum = param.mergeTWNum;

if twSize * twNum > N
    error('twSize * twNum must be no greater than N!');
end

% Get the frame number for block i, frame j (i,j starting from 1)
    function idx = frameIdx(i, j)
        idx = (i - 1) * twSize + j;
    end

imgScale = param.imgScale;

mosaickedTempFile = fullfile(param.resultDir, 'temp_mosaicked.tif');
demosaickedTempFile = fullfile(param.resultDir, 'temp_demosaicked.tif');

if isfield(param, 'cropSize')
    cropSize = param.cropSize;
else
    cropSize = [H, W];
end

% Create temporary CFA file to correct W pixel, if needed
% cfaFile = param.cfaFile;
% if isfield(param, 'wCalibNonneg')
%     wCalib = param.wCalibNonneg;
%     if ~all(wCalib == [1 1 1])
%         cfaFile = fullfile(param.resultDir, 'temp_cfa.tif');
%         [imIdx, cmap] = imread(param.cfaFile);
%         cmap(4, :) = cmap(4, :) .* wCalib;
%         imwrite(imIdx(1:cropSize(1), 1:cropSize(2)), cmap, cfaFile);
%     end
% end
if isfield(param, 'wCalibNonneg')
    wCalib = param.wCalibNonneg;
else
    wCalib = [1 1 1];
end
makeTempCfaFile = true;
cfaFile = fullfile(param.resultDir, 'temp_cfa.tif');
[imIdx, cmap] = imread(param.cfaFile);
cmap(4, :) = cmap(4, :) .* wCalib;
imwrite(imIdx(1:cropSize(1), 1:cropSize(2)), cmap, cfaFile);

%% Sum and compute
ts = tic;
S = zeros(H, W, C);
for i = 1:twNum
    for j = 1:twSize
        S = S + double(imbs{frameIdx(i,j)});
    end
end

% demosaic
im = mleImage(S, twNum * twSize, 1, true);
demosScale = 3 / min(imgScale);
imwrite(im2uint8(im(1:cropSize(1), 1:cropSize(2)) / demosScale), mosaickedTempFile);
command = sprintf('"%s" "%s" "%s" "%s"', ...
    param.demosaicBinPath, cfaFile, mosaickedTempFile, demosaickedTempFile);
status = system(command);
if status ~= 0
    error('Demosaic binary failed. Check command and paths.');
end
im = im2double(imread(demosaickedTempFile)) * demosScale;

if size(im,3) == 1
    im = repmat(im, [1 1 3]);  % duplicate gray into RGB
end
if isscalar(imgScale)
    ima = im(:,:,1:3) * imgScale;  % simple scalar multiplication
elseif numel(imgScale) == 3
    ima = im(:,:,1:3) .* reshape(imgScale, 1, 1, 3);  % per-channel scaling
else
    error('imgScale must be a scalar or length-3 vector');
end
% ima = im(:,:,1:3) .* reshape(imgScale, 1, 1, []);
toc(ts);

% delete temp files
recycle off
delete(mosaickedTempFile);
%delete(demosaickedTempFile);
if makeTempCfaFile
    delete(cfaFile);
end

end
