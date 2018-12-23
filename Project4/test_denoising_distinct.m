% Copyright (c) 2018 Paul Irofti <paul@irofti.net>
% 
% Permission to use, copy, modify, and/or distribute this software for any
% purpose with or without fee is hereby granted, provided that the above
% copyright notice and this permission notice appear in all copies.
% 
% THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
% WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
% MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
% ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
% WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
% ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
% OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

%% DL denoising with training on noisy image
clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
p = 8;                  % patch size
s = 6;                  % sparsity
N = 1000;              % total number of patches
n = 256;                % dictionary size
K = 50;                 % DL iterations
sigma = 20;             % noise standard deviation
%%-------------------------------------------------------------------------
datadir = '../data/savedData/';   % data directory
dataprefix = 'denoise';

imdir = '../data/';      % image directory
img = 'barbara.png'; % original image

addpath(genpath('DL'));
ts = datestr(now, 'yyyymmddHHMMss');
%% Initial Data
errs = zeros(1,K);
fname = [datadir dataprefix '-' img '-sigma' num2str(sigma) ...
    '-n' num2str(n) '-' ts '.mat'];

% Random dictionary
D0 = normc(randn(p^2,n));

% Add noise to original image and vectorize
I = double(imread([imdir,img]));
I = I(:, :, 1);
Inoisy = I + sigma*randn(size(I));

%extract distinct patches
Ynoisy = im2col(Inoisy, [p p], 'distinct');
Ynmean = mean(Ynoisy);
Ynoisy = Ynoisy - repmat(Ynmean,size(Ynoisy,1),1);

save(fname, 'Inoisy', 'Ynoisy', 'Ynmean');

%% Dictionary Learning
%pick N random patched from all patches
Y = Ynoisy(:,randperm(size(Ynoisy,2), N));
D = D0;
for iter = 1:K
    X = omp(Y, D, s);
    [D, X] = aksvd(Y, D, X, iter);
    errs(iter) = norm(Y - D*X, 'fro') / sqrt(numel(Y));
end
save(fname,'D','X','errs','-append');
%% Denoising via Sparse Representation

% Sparse representation
max_s = size(Ynoisy,1)/2;   % maximum density is half the patch
gain = 1.15;                % default noise gain
params = {'error', sqrt(size(Ynoisy,1))*gain*sigma, 'maxatoms', max_s};
Xc = omp(Ynoisy,D,max_s,[],params{:});

% Completati codul Matlab pentru reconstructia imaginii Ic
A = D * Xc + repmat(Ynmean, size(Ynoisy,1), 1);
Ic = col2im(A, [p, p], size(I), 'distinct');
imwrite(uint8(I), 'denoise_I.png');
imwrite(uint8(Inoisy), 'denoise_Inoisy.png');
imwrite(uint8(Ic), 'denoise_Ic.png');

% Clean vs. Original
ipsnr = psnr(Ic, I, 255);
issim = ssim(Ic, I, 'DynamicRange', 255);
sprintf('psnr=%f ssim=%f\n', ipsnr, issim);

save(fname,'Ic','Yc','Xc','ipsnr','issim','-append');
