%% 210116
% Use naiveReconsDemosavariaV1: avoid quantization during demosaicking
% Use lin2rgb
% Save naiveRecons result after hot pixel correction

%% Naive reconstruction with simple averaging
ima = naiveReconsDemosavaria(imbs, param);
imwrite(ima, fullfile(resultDir, 'averageRecons.png'));
imwrite(lin2rgb(ima), fullfile(resultDir, 'averageRecons_g22.png'));
fprintf('Finished naive reconstruction.\n');


imaf = naiveReconsDemosavaria(imbs, param);
% imwrite(imaf, fullfile(resultDir, 'averageReconsHPFixed.png'));
imwrite(lin2rgb(imaf), fullfile(resultDir, 'averageReconsHPFixed_g22.png'));
% save(fullfile(resultDir, 'naiveRecons.mat'), 'ima', 'imaf');
fprintf('Finished naive reconstruction.\n');

%% Align
[flows, flowrs] = patchAlignBinaryDemosavaria(imbs, param);
if param.debug
    save(fullfile(resultDir, 'patchAlign.mat'), 'flows', 'flowrs', 'param');
end
fprintf('Finished alignment.\n');

%% Merge
[Sr, baDemosaicked] = patchMergeBinaryDemosavaria(imbs, flows, param);
paramNoBm3d = param;
paramNoBm3d.bm3dSigma = 0.1;
imr = postMerge(Sr, paramNoBm3d, false);
imwrite(imr, fullfile(resultDir, 'patchMerge.png'));
imwrite(lin2rgb(imr), fullfile(resultDir, 'patchMerge_g22.png'));
save(fullfile(resultDir, 'patchMerge.mat'), 'param', 'Sr', 'imr');
if param.debug
    save(fullfile(resultDir, 'baDemosaicked.mat'), 'baDemosaicked');
end
fprintf('Finished merging.\n');

%% Refine flow and merge
if param.doRefine
    [Srr, barDemosaicked] = patchMergeBinaryDemosavaria(imbs, flowrs, param);
    imrr = postMerge(Srr, paramNoBm3d, false);
    imwrite(imrr, fullfile(resultDir, 'patchMerge_refinedFlow.png'));
    imwrite(lin2rgb(imrr), fullfile(resultDir, 'patchMerge_refinedFlow_g22.png'));
    % save(fullfile(resultDir, 'patchMerge_refinedFlow.mat'), 'param', 'Srr', 'imrr');
    if param.debug
        save(fullfile(resultDir, 'baDemosaicked_refinedFlow.mat'), 'barDemosaicked');
    end
    fprintf('Finished flow refinement.\n');
end

%% Superresolution
if param.doSR
    Ssr = patchDemosaicSR(imbs, flows, param, Sr, baDemosaicked);
    imsr = postMerge(Ssr, paramNoBm3d, true);
    imsr(imsr>1) = 1;
    imsrw = sum(imsr, 3) / 3;
    save(fullfile(resultDir, 'patchRgbSR.mat'), 'imsrw', 'imsr', 'Ssr', 'param');
    imwrite(imsrw, fullfile(resultDir, 'patchRgbSR_w.png'));
    imwrite(lin2rgb(imsrw), fullfile(resultDir, 'patchRgbSR_w_g22.png'));
    imwrite(imsr, fullfile(resultDir, 'patchRgbSR.png'));
    imwrite(lin2rgb(imsr), fullfile(resultDir, 'patchRgbSR_g22.png'));
    
    imsrbm = postMerge(Ssr, param, true);
    save(fullfile(resultDir, 'bm3d.mat'), 'imsrbm', 'param');
    imwrite(imsrbm, fullfile(resultDir, 'bm3d.png'));
    imwrite(lin2rgb(imsrbm), fullfile(resultDir, 'bm3d_g22.png'));
    fprintf('Finished super-resolution.\n');
end

if param.doRefineSR
    Ssrr = patchDemosaicSR(imbs, flowrs, param, Srr, barDemosaicked);
    imsrr = postMerge(Ssrr, paramNoBm3d, true);
    imsrr(imsrr>1) = 1;
    imsrrw = sum(imsrr, 3) / 3;
    save(fullfile(resultDir, 'patchRgbSR_refinedFlow.mat'), 'imsrrw', 'imsrr', 'Ssr', 'param');
    imwrite(imsrrw, fullfile(resultDir, 'patchRgbSR_w_refinedFlow.png'));
    imwrite(lin2rgb(imsrrw), fullfile(resultDir, 'patchRgbSR_w_refinedFlow_g22.png'));
    imwrite(imsrr, fullfile(resultDir, 'patchRgbSR_refinedFlow.png'));
    imwrite(lin2rgb(imsrr), fullfile(resultDir, 'patchRgbSR_refinedFlow_g22.png'));
    
    imsrrbm = postMerge(Ssrr, param, true);
    save(fullfile(resultDir, 'bm3d_refinedFlow.mat'), 'imsrrbm', 'param');
    imwrite(imsrrbm, fullfile(resultDir, 'bm3d_refinedFlow.png'));
    imwrite(lin2rgb(imsrrbm), fullfile(resultDir, 'bm3d_refinedFlow_g22.png'));
    fprintf('Finished super-resolution with flow refinement.\n');
end

%% Compute PSNR
if param.computePSNR
    result = struct('param', param);
    [H, W, ~] = size(imr);
    imgtr = imresize(imgt, [H W]);
    imgtr(imgtr>1) = 1;
    imgtr(imgtr<0) = 0;
    psnr = evalPSNR(ima, imgtr);
    result.quantaNaivePSNR = psnr;
    psnr = evalPSNR(imr, imgtr);
    result.quantaBurstPSNR = psnr;
    
    if param.doRefine
        psnr = evalPSNR(imrr, imgtr);
        result.quantaBurstRefinedPSNR = psnr;
    end
    
    if param.doSR
        [Hsr, Wsr, ~] = size(imsr);
        imgtsr = imresize(imgt, [Hsr Wsr]);
        psnr = evalPSNR(imsrbm, imgtsr);
        result.quantaBurstSrPSNR = psnr;
    end
    
    if param.doRefineSR
        [Hsr, Wsr, ~] = size(imsr);
        imgtsr = imresize(imgt, [Hsr Wsr]);
        psnr = evalPSNR(imsrrbm, imgtsr);
        result.quantaBurstSrRefinedPSNR = psnr;
    end
    
    savejson('', result, fullfile(resultDir, 'quanta.json'));
end