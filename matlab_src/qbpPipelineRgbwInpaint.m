function [result] = qbpPipelineRgbwInpaint(imbs, param, imgt)
%QBPPIPELINEMONO Entire CQBP pipeline for rgbw binary images
% Use inpainting to fill in W channel
% based on srcRgbwSrInpaint_220226

result = struct();
resultDir = param.resultDir;

%% Naive reconstruction with simple averaging
ima = naiveReconsDemosavaria(imbs, param);
if param.imgAutoScale
    [ima, param.imgScale] = autoScaleIntensity(ima, 97);
end
asParam = param;
asParam.mergeTWNum = 1;
refBlock = floor((param.refFrame - 1) / param.mergeTWSize);
imas = naiveReconsDemosavaria(imbs(refBlock*param.mergeTWSize+1:(refBlock+1)*param.mergeTWSize), asParam);

if param.saveImages
    imwriteResult(ima, 'averageRecons', param);
    imwriteResult(imas, 'averageReconsShort', param);
end
fprintf('Finished naive reconstruction.\n');

%% Align
[flows, flowrs] = patchAlignBinaryInpaintW(imbs, param);
if param.debug
    save(fullfile(resultDir, 'patchAlign.mat'), 'flows', 'flowrs', 'param');
end
result.flows = flows;
result.flowrs = flowrs;
fprintf('Finished alignment.\n');

%% Merge
[Sr, baDemosaicked] = patchMergeBinaryInpaintW(imbs, flows, param);
imr = postMerge(Sr, param, false);
result.Sr = Sr;
result.baDemosaicked = baDemosaicked;
result.imr = imr;
if param.saveImages
    imwriteResult(imr, 'patchMerge', param);
end
if param.debug
    save(fullfile(resultDir, 'patchMerge.mat'), 'param', 'Sr', 'imr');
    save(fullfile(resultDir, 'baDemosaicked.mat'), 'baDemosaicked');
end
fprintf('Finished merging.\n');

%% Refine flow and merge
if param.doRefine
    [Srr, barDemosaicked] = patchMergeBinaryInpaintW(imbs, flowrs, param);
    imrr = postMerge(Srr, param, false);
    result.Srr = Srr;
    result.barDemosaicked = barDemosaicked;
    result.imrr = imrr;
    if param.saveImages
        imwriteResult(imrr, 'patchMerge_refinedFlow', param);
    end
    if param.debug
        save(fullfile(resultDir, 'patchMerge_refinedFlow.mat'), 'param', 'Srr', 'imrr');
        save(fullfile(resultDir, 'baDemosaicked_refinedFlow.mat'), 'barDemosaicked');
    end
    fprintf('Finished flow refinement.\n');
end

%% Superresolution & demosaic
if param.doSR
    Ssr = patchDemosaicSR(imbs, flows, param, Sr, baDemosaicked);
    imsrnf = postMerge(Ssr(:,:,1:3), param, true);
    [imsr, imsrw] = postDemosaicSrRgbw(Ssr, param);
%     imsr(imsr>1) = 1;

    imsrbm = postDemosaicSrRgbwBm3d(Ssr, param);
    
    % Further adjust scaling for best visual quality
    if param.imgAutoScale
        [imsrbm, imgScaleCorrect] = autoScaleIntensity(imsrbm, 97);
        param.imgScale = param.imgScale * imgScaleCorrect;
        imsrnf = imsrnf * imgScaleCorrect;
        imsr = imsr * imgScaleCorrect;
        imsrw = imsrw * imgScaleCorrect;
        
        % Recompute naive images with right intensity range
        avgParam = param;
        avgParam.imgAutoScale = false;
        ima = naiveReconsDemosavaria(imbs, avgParam);
        avgParam.mergeTWNum = 1;
        refBlock = floor((param.refFrame - 1) / param.mergeTWSize);
        imas = naiveReconsDemosavaria(imbs(refBlock*param.mergeTWSize+1:(refBlock+1)*param.mergeTWSize), avgParam);
    end
    
    if param.saveImages
        imwriteResult(ima, 'averageRecons', param);
        imwriteResult(imas, 'averageReconsShort', param);
        imwriteResult(imsrnf, 'patchRgbSR_nofix', param, true);
        imwriteResult(imsr, 'patchRgbSR', param, true);
        imwriteResult(imsrw, 'patchRgbSR_w', param, true);
        imwriteResult(imsrbm, 'qbm3d', param, true);
    end
    if param.debug
        save(fullfile(resultDir, 'naiveRecons.mat'), 'ima', 'imas');
        save(fullfile(resultDir, 'patchRgbSR.mat'), 'imsrw', 'imsr', 'Ssr', 'param');
        save(fullfile(resultDir, 'qbm3d.mat'), 'imsrbm', 'param');
    end
    result.ima = ima;
    result.imas = imas;
    result.Ssr = Ssr;
    result.imsrnf = imsrnf;
    result.imsr = imsr;
    result.imsrw = imsrw;
    result.imsrbm = imsrbm;
    result.param = param;

    fprintf('Finished super-resolution.\n');
end

if param.doRefineSR
    Ssrr = patchDemosaicSR(imbs, flowrs, param, Srr, barDemosaicked);
    imsrrnf = postMerge(Ssrr(:,:,1:3), param, true);
    [imsrr, imsrrw] = postDemosaicSrRgbw(Ssrr, param);
%      imsrr(imsrr>1) = 1;

    imsrrbm = postDemosaicSrRgbwBm3d(Ssr, param);

    if param.saveImages
        imwriteResult(imsrrnf, 'patchRgbSR_nofix_refinedFlow', param, true);
        imwriteResult(imsrrw, 'patchRgbSR_w_refinedFlow', param, true);
        imwriteResult(imsrr, 'patchRgbSR_refinedFlow', param, true);
        imwriteResult(imsrrbm, 'qbm3d_refinedFlow', param, true);
    end
    if param.debug
        save(fullfile(resultDir, 'patchRgbSR_refinedFlow.mat'), 'imsrrw', 'imsrr', 'Ssr', 'param');
        save(fullfile(resultDir, 'qbm3d_refinedFlow.mat'), 'imsrrbm', 'param');
    end

    result.Ssrr = Ssrr;
    result.imsrrnf = imsrrnf;
    result.imsrrw = imsrrw;
    result.imsrr = imsrr;
    result.imsrrbm = imsrrbm;
    
    fprintf('Finished super-resolution with flow refinement.\n');
end

%% Compute PSNR
% Need to turn off imgAutoScale and set imgScale=1?
% if param.computePSNR
%     result = struct('param', param);
%     [H, W, ~] = size(imr);
%     imgtr = imresize(imgt, [H W]);
%     imgtr(imgtr>1) = 1;
%     imgtr(imgtr<0) = 0;
%     psnr = evalPSNR(ima, imgtr);
%     result.quantaNaivePSNR = psnr;
%     psnr = evalPSNR(imr, imgtr);
%     result.quantaBurstPSNR = psnr;
%     
%     if param.doRefine
%         psnr = evalPSNR(imrr, imgtr);
%         result.quantaBurstRefinedPSNR = psnr;
%     end
%     
%     if param.doSR
%         [Hsr, Wsr, ~] = size(imsrbm);
%         imgtsr = imresize(imgt, [Hsr Wsr]);
%         psnr = evalPSNR(imsrbm, imgtsr);
%         result.quantaBurstSrPSNR = psnr;
%     end
%     
%     if param.doRefineSR
%         [Hsr, Wsr, ~] = size(imsrrbm);
%         imgtsr = imresize(imgt, [Hsr Wsr]);
%         psnr = evalPSNR(imsrrbm, imgtsr);
%         result.quantaBurstSrRefinedPSNR = psnr;
%     end
%     
%     savejson('', result, fullfile(resultDir, 'quanta.json'));
% end

end

