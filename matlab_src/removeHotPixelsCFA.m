function imb = removeHotPixelsCFA(imb, dcr, thresh, cfaFile)
%REMOVEHOTPIXELSCFA Remove hot pixels from binary images by interpolating from
%surrounding pixels (with the same color filter)

% Determine the hot pixel mask
hpInd = find(dcr > thresh);
hpCount = numel(hpInd);

% Load cfa index file
[folder, file, ~] = fileparts(cfaFile);
cfaIndexFile = fullfile(folder, [file '.tif']);
load(cfaIndexFile, 'cfaKnn');
[N, k] = size(cfaKnn);
    
% Process each frame
if iscell(imb)
    for i = 1:numel(imb)
        replaceInd = cfaKnn(sub2ind([N, k], hpInd, randi([1, k], [hpCount 1])));
        imb{i}(hpInd) = imb{i}(replaceInd);
    end
else
    replaceInd = cfaKnn(sub2ind([N, k], hpInd, randi([1, k], [hpCount 1])));
    imb(hpInd) = imb(replaceInd);
end

