function [result] = qbpPipelineIntensity(ims, param)
%QBPPIPELINEINTENSITY Entire QBP pipeline for intensity images
% ims: cell array of grayscale intensity images
% param: QBP parameters

result = struct();
resultDir = param.resultDir;

%% Pad image to power of 2
H = size(ims{1}, 1);
W = size(ims{1}, 2);
newH = 2^ceil(log2(H));
newW = 2^ceil(log2(W));
padU = floor((newH - H) / 2);
padD = newH - H - padU;
padL = floor((newW - W) / 2);
padR = newW - W - padL;
imp = cell(1, numel(ims));
for i = 1:numel(ims)
    imtemp = padarray(ims{i}, [padU, padL], 'replicate', 'pre');
    imtemp = padarray(imtemp, [padD, padR], 'replicate', 'post');
    imp{i} = imtemp;
end

%% Align
flows = patchAlign(imp, param, imp);
result.flows = flows;
fprintf('Finished alignment.\n');

%% Merge
imm = patchMergeIntensity(imp, flows, param);
imm = imm(1+padU:end-padD, 1+padL:end-padR, :);

if param.saveImages
    imwrite(imm, fullfile(resultDir, 'merge_result.png'));
end
if param.debug
    save(fullfile(resultDir, 'merge_result.mat'), 'param', 'flows', 'imm');
end
result.imm = imm;
result.param = param;
fprintf('Finished merging.\n');

%% Refine flow and merge
if param.doRefine
    flowrs = patchAlignRefine(imp, flows, param, imp);
    immr = patchMergeIntensity(imp, flowrs, param);
    immr = immr(1+padU:end-padD, 1+padL:end-padR, :);
    if param.debug
        save(fullfile(resultDir, 'merge_refined_result.mat'), 'param', 'immr', 'flowrs');
    end
    if param.saveImages
        imwrite(immr, fullfile(resultDir, 'merge_refined_result.png'));
    end
    result.immr = immr;
    fprintf('Finished flow refinement.\n');
end


end

