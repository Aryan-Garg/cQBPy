function S = patchDemosaicSR(imbs, flows, param, mergeSumImage, baDemosaicked)
%PATCHDEMOSAICSR Super-resolution demosaicking from binary images
% using Google's algorithm, which works for both bayer and random CFA
% works for any number of filters, should be followed by a color conversion
% V7:  
% 210226: add support for patch{Align,Merge}BinaryDownsampleW
%Input:
%  imbs: 1D cell array of binary frames
%  flows: patchwise flow fields for keyframes
%  param: struct that contains following fields:
%    patchSize: spatial patch size for wiener filtering (divisible)
%    alignTWSize: temporal window size for alignment
%    alignTWNum: number of temporal windows for alignment
%    mergeTWSize: temporal window size for merging
%    mergeTWNum: number of temporal windows for merging, total number of
%                frames used for merging <= for alignment
%    srTWSize: temporal window size for SR
%    srTWNum: number of temporal windows for SR, total number of
%                frames used for merging <= for alignment
%    refFrame: reference frame #
%    cfa: color filter array, 'rggb', 'bggr', 'grbg' or 'gbrg'
%    cfaFile: indexed color image for cfa
%    srScale: output scale, outputSize = ceil(originalSize*srScale)
%    combineRadius: radius of the neighborhood to look up for merging
%    s1: scale factor for the robustness term, for motion > M_th
%    s2: scale factor for the robustness term, for motion <= M_th
%    sc: additional parameter to scale the standard deviation for the
%        robustness term
%    M_th: threshold for local motion
%    M_sigma: sigma for local motion discontinuity
%    t: threshold for the robustness term
%    D_tr: denoising decreasing rate
%    D_th: denoising threshold
%    k_detail: base kernel standard deviation
%    k_stretch: amount of kernel stretching along edges
%    k_shrink: amount of kernel shrinking perpendicular to edges
%    k_denoise: denoising kernel standard deviation
%    debug: whether or not to print debug information
%  mergeSumImage: result of normal merging, used as guide image
%  baDemosaicked: block aggregations, shape = (H, W, C, mergeTWNum)
%                 C can be either 1 (interpolated W) or 3 (demosaicked)
%Output:
%  imr: reconstructed image

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
assert(C == 1); % inputs must be mosaicked images
srScale = param.srScale;
Ho = ceil(H*srScale);
Wo = ceil(W*srScale);
patchSize = param.patchSizes(1);
alignTWSize = param.alignTWSize;
alignTWNum = param.alignTWNum;
srTWSize = param.srTWSize;
srTWNum = param.srTWNum;

if srTWSize * srTWNum > alignTWSize * alignTWNum
    error('srTWSize * srTWNum must be no greater than alignTWSize * alignTNNum!');
end

refFrame = param.refFrame;
if srTWSize * srTWNum < refFrame
    error('srTWSize * srTWNum must be no smaller than refFrame!');
end
cenFrame = mod(refFrame - 1, srTWSize) + 1; 

% Get the frame number for block i, frame j (i,j starting from 1)
    function idx = frameIdx(i, j)
        idx = (i - 1) * srTWSize + j;
    end

% Get the align block and frame subscripts for a give nframe no.
    function [i, j] = alignSub(idx)
        i = floor((idx-1)/alignTWSize)+1;
        j = mod(idx-1, alignTWSize) + 1;
    end

if nargin < 4
    mergeSumImage = [];
end

% generate the CFA array for later processing
if isfield(param, 'cfaFile') && ~isempty(param.cfaFile)
    [cfaIdx, cfaMap] = imread(param.cfaFile);
    assert(~isempty(cfaMap)) % confirm the CFA image is an indexed image
    cfaIdx = cfaIdx + 1; % cmap idx starts from 0
    numC = max(cfaIdx, [], 'all');
    % TODO: to expand baDemosaicked to all filters?
%     if numC > 3
%         baDemosaicked = cat(3, baDemosaicked,...
%             zeros(size(baDemosaicked,1), size(baDemosaicked,2), numC-3, size(baDemosaicked,4)));
%         for c = 4:numC
%             baDemosaicked(:,:,c,:) = baDemosaicked(:,:,1,:) * cfaMap(c, 1)...
%                 + baDemosaicked(:,:,2,:) * cfaMap(c, 2)...
%                 + baDemosaicked(:,:,3,:) * cfaMap(c, 3);
%         end
%     end
elseif ischar(param.cfa)
    switch param.cfa
        case 'rggb'
            cfaIdx = [1 2; 2 3];
        case 'bggr'
            cfaIdx = [3 2; 2 1];
        case 'grbg'
            cfaIdx = [2 1; 3 2];
        case 'gbrg'
            cfaIdx = [2 3; 1 2];
    end
    numC = 3;
else
    error('param.cfa: array type not supported yet!');
%     cfaIdx = param.cfa(:,:,1);
%     for c = 2:size(param.cfa, 3)
%         cfaIdx = cfaIdx + param.cfa(:,:,c)*c;
%     end
end
assert(mod(H,size(cfaIdx,1)) == 0 && mod(W,size(cfaIdx,2)) == 0);
cfaIdx = repmat(cfaIdx, H/size(cfaIdx,1), W/size(cfaIdx,2));

%% Merge
% "reference frame" in each align block
alignCenFrame = mod(refFrame - 1, alignTWSize) + 1; 

% Preprocess the flows for interpolation
flowsr = cell(1, alignTWNum+2);
skip = 2; % assumes patchSize = 2 * patchStride
for i = 1:alignTWNum
    flowsr{i+1} = flows{i}(1:skip:end,1:skip:end,:);
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

% Phase 1: Compute the robustness weight
% Compute the LR reconstruction images for blocks, 
% warped using cubic, only used for computing R
% TODO: when a mergeSumImage is not available?
[X, Y] = meshgrid(1:W, 1:H);
blockAvg = cell(1, param.mergeTWNum);
fprintf('Computing LR reconstruction images...\n');
% TODO: weight computed on merge block level. Is this OK?
for i = 1:param.mergeTWNum
    curFlow = flowsr{i+1};
    flowwarp = repelem(curFlow, patchSize, patchSize, 1);
    destX = X + round(flowwarp(:,:,1));
    destY = Y + round(flowwarp(:,:,2));
    if size(baDemosaicked, 3) == 1
        blockAvg{i} = interp2(baDemosaicked(:,:,1,i), destX, destY, 'cubic');
    else
        assert(size(baDemosaicked, 3) == 3);
        temp = zeros(H, W, 3);
        for c = 1:3
            temp(:,:,c) = interp2(baDemosaicked(:,:,c,i), destX, destY, 'cubic');
        end
        temp = mleImage(temp*param.mergeTWSize, param.mergeTWSize, 1, true);
        temp = mean(temp, 3);
        temp = mleImageInv(temp, param.mergeTWSize, 1);
        temp = temp / param.mergeTWSize;
        blockAvg{i} = temp;
    end
end
% for i = 1:srTWNum
%     Sb = zeros(H, W);
%     countMap = zeros(H, W);
%     for j = 1:srTWSize
%         curFlow = interpFlow(i, j);
%         flowwarp = repelem(curFlow, patchSize, patchSize, 1);
%         destX = X + round(flowwarp(:,:,1));
%         destY = Y + round(flowwarp(:,:,2));
%         temp = interp2(sum(double(imbs{frameIdx(i,j)}),3), destX, destY, 'cubic');
%         countMap = countMap + isfinite(temp);
%         temp(~isfinite(temp)) = 0;
%         Sb = Sb + temp;
%     end
%     countMap(countMap == 0) = 1;
% %     blockMerge{i} = mleIntensity(Sb, countMap*C, tau, eta) / maxFlux;
%     blockMerge{i} = Sb ./ countMap / C;
% end

if size(mergeSumImage, 3) == 1
    baGuide = mergeSumImage / param.mergeTWSize / param.mergeTWNum;
else
    mergeImage = mleImage(mergeSumImage, param.mergeTWSize*param.mergeTWNum, 1, true);
    mergeImageBW = mean(mergeImage, 3);
    baGuide = mleImageInv(mergeImageBW, param.mergeTWSize*param.mergeTWNum, 1);
    baGuide = baGuide / param.mergeTWSize / param.mergeTWNum;
end

% TODO: to extend to numC channels?
% if numC > 3
%     imguideColor = cat(3, imguideColor,...
%         zeros(size(imguideColor, 1), size(imguideColor, 2), numC-3));
%     for c = 4:numC
%         imguideColor(:,:,c) = imguideColor(:,:,1) * cfaMap(c, 1)...
%             + imguideColor(:,:,2) * cfaMap(c, 2)...
%             + imguideColor(:,:,3) * cfaMap(c, 3);
%     end
% end

% Compute the weight term, shared for all color channels
% TODO: should I compute R for each individual channel?
% TODO: add time dependent weight?
fprintf('Phase 1...\n');
ts = tic;
rbs_sigma_ms = stdfilt(baGuide, true(3));
SguideMean = imfilter(baGuide, ones(3)/9);
% from binomial statistics
% rbs_sigma_md = sqrt(SguideMean .* (1-SguideMean) / param.mergeTWSize / param.mergeTWNum);
if isfield(param, 'cfaMethod') && strcmp(param.cfaMethod, 'downsampleW')
    cfaMask = double(cfaIdx == 4);
    cfaT = boxDownsample(cfaMask, param.cfaAtomSize, false);
    cfaTUS = repelem(cfaT, param.cfaAtomSize, param.cfaAtomSize);
    sampleNum = param.mergeTWSize * cfaTUS;
else
    sampleNum = param.mergeTWSize;
end
rbs_sigma_md = sqrt(SguideMean .* (1-SguideMean) ./ sampleNum);
% rbs_sigma = max(rbs_sigma_ms, rbs_sigma_md);
Rm = zeros(H, W, param.mergeTWNum);
for i = 1:param.mergeTWNum
    rbs_d_ms = abs(blockAvg{i} - SguideMean);
%     rbs_d_md = sqrt(blockMerge{i} .* (1-blockMerge{i}) / param.mergeTWSize);
%     rbs_d = rbs_d_ms .* rbs_d_ms.^2 ./ (rbs_d_ms.^2 + rbs_d_md.^2);
    R0 = exp(-rbs_d_ms.^2./ (param.sc^2 * (rbs_sigma_ms.^2 + rbs_sigma_md.^2)));
%     R0 = exp(-rbs_d.^2./rbs_sigma.^2);
    
    Mx = imdilate(flowsr{i+1}(:,:,1), ones(3)) - imerode(flowsr{i+1}(:,:,1), ones(3));
    My = imdilate(flowsr{i+1}(:,:,2), ones(3)) - imerode(flowsr{i+1}(:,:,2), ones(3));
    M = sqrt(Mx.^2 + My.^2);
    
%     s_M = max(0.3, exp(-M / param.M_sigma));
%     s_M = repmat(s_M, patchSize, patchSize);
%     R(:,:,i) = exp(-rbs_d.^2./(s_M.*rbs_sigma).^2);
    
    s = ones(size(M)) * param.s2;
    s(M>param.M_th) = param.s1;
    s = imresize(s, [H W], 'nearest');
    Rm(:,:,i) = min(max(s .* R0 - param.t, 1e-2), 1);
    Rm(:,:,i) = imerode(Rm(:,:,i), ones(5));
end
% Broadcast to sr blocks
srIdx = (0:srTWNum-1)*srTWSize;
mergeIdx = floor(srIdx / param.mergeTWSize);
mergeRe = mod(srIdx, param.mergeTWSize);
R = zeros(H, W, srTWNum);
for i = 1:srTWNum
    spillover = mergeRe(i) + srTWSize - param.mergeTWSize;
    if spillover <= 0
        R(:,:,i) = Rm(:,:,mergeIdx(i)+1);
    else
        R(:,:,i) = Rm(:,:,mergeIdx(i)+1) * (1-spillover/param.mergeTWSize)...
            + Rm(:,:,mergeIdx(i)+2) * spillover/param.mergeTWSize;
    end
end
toc(ts);

% % TODO: or for numC channels? or only 3 channels and them
% % convert to numC channels?
% fprintf('Phase 1...\n');
% ts = tic;
% rbs_sigma_ms = stdfilt(imguideColor, true(3));
% mergeSumMean = imfilter(imguideColor, ones(3)/9);
% rbs_sigma_md = sqrt(mergeSumMean .* (1-mergeSumMean) / param.mergeTWSize); % need to double-check these equations
% rbs_sigma = max(rbs_sigma_ms, rbs_sigma_md);
% rbs_d_md = rbs_sigma_md;
% R = zeros(H, W, 3, param.mergeTWNum);
% for i = 1:param.mergeTWNum
%     rbs_d_ms = abs(blockMerge{i} - mergeSumMean);
%     rbs_d = rbs_d_ms .* rbs_d_ms.^2 ./ (rbs_d_ms.^2 + rbs_d_md.^2);
%     R0 = exp(-rbs_d.^2./rbs_sigma.^2);
%     
%     Mx = imdilate(flowsr{i+1}(:,:,1), ones(3)) - imerode(flowsr{i+1}(:,:,1), ones(3));
%     My = imdilate(flowsr{i+1}(:,:,2), ones(3)) - imerode(flowsr{i+1}(:,:,2), ones(3));
%     M = sqrt(Mx.^2 + My.^2);
%     s = ones(size(M)) * param.s2;
%     s(M>param.M_th) = param.s1;
%     s = imresize(s, [H W], 'bilinear');
%     R(:,:,:,i) = min(max(s .* R0 - param.t, 0), 1);
% %     R(:,:,i) = imerode(R(:,:,i), ones(5));
% end
% R = repelem(R, 1, 1, 1, param.mergeTWSize);
% toc(ts);

% Phase 2: warp the block-sum images using nearest
% record the warped block-sum images and the subpixel X, Y offset
fprintf('Phase 2...\n');
ts = tic;
N = srTWNum;
warpS = zeros(H, W, N);
warpCfaIdx = zeros(H, W, N);
warpX = zeros(H, W, N);
warpY = zeros(H, W, N);
for i = 1:srTWNum
    blockAggre = zeros(H, W);
    for j = 1:srTWSize
        blockAggre = blockAggre + imbs{frameIdx(i,j)};
    end
    curFlow = interpFlow(i, cenFrame);
    flowwarp = repelem(curFlow, patchSize, patchSize, 1);
    destX = X + round(flowwarp(:,:,1));
    destY = Y + round(flowwarp(:,:,2));
    temp = interp2(blockAggre, destX, destY, 'nearest');
    nanMask = isnan(temp);
    temp(nanMask) = 0;
    warpS(:,:,i) = temp;
    temp = interp2(cfaIdx, destX, destY, 'nearest');
    temp(nanMask) = NaN;
    warpCfaIdx(:,:,i) = temp;
    temp = destX - flowwarp(:,:,1);
    temp(nanMask) = NaN;
    warpX(:,:,i) = temp;
    temp = destY - flowwarp(:,:,2);
    temp(nanMask) = NaN;
    warpY(:,:,i) = temp;
    if param.debug
        fprintf('.');
    end
end
if param.debug
    fprintf('\n');
end
toc(ts);

% Phase 3: compute the pixel values at each pixel at HR grid,
% for all color channels
fprintf('Phase 3...\n');
ts = tic;
S = zeros(Ho, Wo, numC);
% Compute the structure tensors
[Ix, Iy] = imgradientxy(baGuide, 'sobel');
Ix = Ix/8; Iy = Iy/8; % normalize
Ixx = Ix .^ 2; Ixy = Ix .* Iy; Iyy = Iy .^ 2;
STa = imfilter(Ixx, ones(3)); STb = imfilter(Ixy, ones(3)); STd = imfilter(Iyy, ones(3));
STT = STa + STd; STD = STa .* STd - STb .^ 2;
lambda1 = STT/2 + sqrt(STT.^2/4-STD);
lambda2 = STT - lambda1;
A = min(sqrt(lambda1./lambda2), 5);
A(isnan(A)) = 1;
D = 1 - sqrt(lambda1)/param.D_tr + param.D_th;
D(D<0) = 0; D(D>1) = 1;
% notice: different to orignal equation in Wronski et al.
% k_detail not multiplied here (multiplied later to Omega)
k1hat = 1 * param.k_stretch * A;
k2hat = 1 ./ (param.k_shrink * A);
k1 = ((1-D).*k1hat + D*1*param.k_denoise).^2;
k2 = ((1-D).*k2hat + D*1*param.k_denoise).^2;
if numel(param.k_detail) == 1
    k_detail = repmat(param.k_detail, numC, 1);
else
    k_detail = param.k_detail;
end

fprintf('Evaluate Pixels...\n');
iOffset = (1+H)/2 - (1+Ho)/2/srScale;
jOffset = (1+W)/2 - (1+Wo)/2/srScale;
for i = 1:Ho
    if param.debug
        fprintf('%d', i);
    end
    for j = 1:Wo
        accC = zeros(1, 1, numC);
        accW = zeros(1, 1, numC);
        iOrig = iOffset + i/srScale; % pixel index in original grid
        jOrig = jOffset + j/srScale;
        iOrigR = round(iOrig);
        jOrigR = round(jOrig);
        % Compute the covariance matrix Omega
        if STb(iOrigR,jOrigR) == 0
            if STa(iOrigR,jOrigR) > STd(iOrigR,jOrigR)
                e1 = [1; 0];
                e2 = [0; 1];
            else
                e1 = [0; 1];
                e2 = [1; 0];
            end
        else
            e1 = [STb(iOrigR,jOrigR); lambda1(iOrigR,jOrigR)-STa(iOrigR,jOrigR)];
            e1 = e1 / norm(e1);
            e2 = [STb(iOrigR,jOrigR); lambda2(iOrigR,jOrigR)-STa(iOrigR,jOrigR)];
            e2 = e2 / norm(e2);
        end
        % Omega = [e1 e2]*[k1 0;0 k2]*[e1 e2]';
        for c = 1:numC
            OmegaInv{c} = [e2 e1]*[1/k1(iOrigR,jOrigR) 0; 0 1/k2(iOrigR,jOrigR)]*[e2 e1]'...
                /k_detail(c)^2; %#ok<AGROW>
        end
        % iterate over all samples within combineRadius from all blocks
        for k = max(1,iOrigR-param.combineRadius):min(H,iOrigR+param.combineRadius)
            for l = max(1,jOrigR-param.combineRadius):min(W,jOrigR+param.combineRadius)
                d = [reshape(warpX(k,l,:)-jOrig, 1, []); reshape(warpY(k,l,:)-iOrig, 1, [])];
                % proces R, G, B separately
                for c = 1:numC
                    w0 = exp(-sum(OmegaInv{c}*d.*d,1)/2);
                    w = w0 .* reshape(R(k,l,:),1,[]);
                    w(isnan(w)) = 0;
                    colorMask = reshape(warpCfaIdx(k,l,:) == c, 1, []);
                    ws = w .* reshape(warpS(k,l,:),1,[]);
                    accC(c) = accC(c) + sum(ws(colorMask));
                    accW(c) = accW(c) + sum(w(colorMask));
                end
            end
        end
        S(i,j,:) = max(accC ./ accW, 0);
        if param.debug
            fprintf('.');
        end
    end
    if param.debug
        fprintf('\n');
    end
end
toc(ts);

% S = mleImage(S*srTWNum*srTWSize, srTWNum*srTWSize, imgScale);
S = S*srTWNum;
fprintf('Super-resolution done. ');
toc(ts0);

end
