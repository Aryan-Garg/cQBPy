function out = python_parity_pipeline(dataMatPath, outMatPath)
% Minimal MATLAB/Octave parity reference for Python pipeline
% Input MAT contains: imbs (H,W,T), imgt (H,W,3), searchRadii, refFrame, tau, eta, dcr

s = load(dataMatPath);
imbs = double(s.imbs);
imgt = double(s.imgt);
searchRadii = double(s.searchRadii);
refFrame = double(s.refFrame);
tau = double(s.tau);
eta = double(s.eta(:)');
dcr = double(s.dcr);

[H, W, T] = size(imbs);
refIdx = min(max(refFrame+1, 1), T);

% naive reconstruction
avg = mean(imbs, 3);
rgb_prob = demosaic_bayer_bilinear_local(avg);
naive = post_merge_local(rgb_prob, T, tau, eta, dcr);

% global translational alignment
flows = zeros(H, W, 2, T);
ref = imbs(:,:,refIdx);
for i = 1:T
    if i == refIdx
        continue;
    end
    src = imbs(:,:,i);
    best_cost = inf;
    best_dx = 0; best_dy = 0;
    for r = searchRadii
        for dy = -r:r
            for dx = -r:r
                warped = shift_nearest(src, dy, dx);
                cost = mean(abs(warped(:) - ref(:)));
                if cost < best_cost
                    best_cost = cost;
                    best_dx = dx; best_dy = dy;
                end
            end
        end
    end
    flows(:,:,1,i) = best_dx;
    flows(:,:,2,i) = best_dy;
end

% robust merge approximation (same as Python fallback: flow-warp then mean per CFA color)
rgb_prob_merge = zeros(H,W,3);
count = zeros(H,W,3);
for i = 1:T
    dx = flows(1,1,1,i);
    dy = flows(1,1,2,i);
    warped = shift_nearest(imbs(:,:,i), dy, dx);
    mR = false(H,W); mG = false(H,W); mB = false(H,W);
    mR(1:2:end,1:2:end) = true;
    mG(1:2:end,2:2:end) = true;
    mG(2:2:end,1:2:end) = true;
    mB(2:2:end,2:2:end) = true;
    rgb_prob_merge(:,:,1) = rgb_prob_merge(:,:,1) + warped .* mR;
    rgb_prob_merge(:,:,2) = rgb_prob_merge(:,:,2) + warped .* mG;
    rgb_prob_merge(:,:,3) = rgb_prob_merge(:,:,3) + warped .* mB;
    count(:,:,1) = count(:,:,1) + mR;
    count(:,:,2) = count(:,:,2) + mG;
    count(:,:,3) = count(:,:,3) + mB;
end
rgb_prob_merge = rgb_prob_merge ./ max(count, 1);
merged = post_merge_local(rgb_prob_merge, T, tau, eta, dcr);

out = struct();
out.naive = naive;
out.merged = merged;
out.flows = flows;
out.psnr_naive = psnr_local(naive, imgt);
out.psnr_merged = psnr_local(merged, imgt);

if nargin >= 2
    save(outMatPath, '-struct', 'out', '-v7');
end

end

function shifted = shift_nearest(img, dy, dx)
[H, W] = size(img);
[xg, yg] = meshgrid(1:W,1:H);
xs = min(max(round(xg - dx),1),W);
ys = min(max(round(yg - dy),1),H);
idx = sub2ind([H,W], ys, xs);
shifted = img(idx);
end

function rgb = demosaic_bayer_bilinear_local(m)
[H,W] = size(m);
kr = [0 1 0; 1 4 1; 0 1 0] / 4;
kg = [0 1 0; 1 4 1; 0 1 0] / 4;
kb = [0 1 0; 1 4 1; 0 1 0] / 4;
mR = zeros(H,W); mG = zeros(H,W); mB = zeros(H,W);
mR(1:2:end,1:2:end)=1;
mG(1:2:end,2:2:end)=1; mG(2:2:end,1:2:end)=1;
mB(2:2:end,2:2:end)=1;
R = conv2(m.*mR, kr, 'same');
G = conv2(m.*mG, kg, 'same');
B = conv2(m.*mB, kb, 'same');
rgb = cat(3, R,G,B);
end

function rgb = post_merge_local(rgb_prob, total_frames, tau, eta, dcr)
rgb = -log(max(1-rgb_prob, 1e-8)) / tau;
for c = 1:3
    rgb(:,:,c) = (rgb(:,:,c) - dcr) / max(eta(c),1e-8);
end
rgb = rgb / (max(rgb(:)) + 1e-8);
rgb = min(max(rgb, 0), 1);
end

function v = psnr_local(a,b)
mse = mean((a(:)-b(:)).^2);
if mse <= 0
    v = inf;
else
    v = 20*log10(1/sqrt(mse));
end
end
