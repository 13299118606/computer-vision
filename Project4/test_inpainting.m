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

%% DL inpainting with training on masked image
clear; clc; close all; fclose all; format compact;
%%-------------------------------------------------------------------------
p = 8;                  % patch size
s = 6;                  % sparsity
n = 256;                % dictionary size
K = 50;                 % DL iterations
miss = 0.5;             % missing data ratio
%%-------------------------------------------------------------------------
datadir = '../data/savedData/';   % data directory
dataprefix = 'inpaint';

imdir = '../data/';      % image directory
img = 'barbara.png'; % original image

addpath(genpath('DL'));
ts = datestr(now, 'yyyymmddHHMMss');
%% Initial Data
errs = zeros(1,K);
fname = [datadir dataprefix '-' img '-ratio' num2str(miss) ...
    '-n' num2str(n) '-' ts '.mat'];

% Random dictionary
D0 = normc(randn(p^2,n));

% Create missing data mask and vectorize
I = double(imread([imdir,img]));
I = I(:, :, 1);
Y = im2col(I, [p p], 'distinct');
Ymean = mean(Y);
Y = Y - repmat(Ymean,size(Y,1),1);

Ymask = rand(size(Y));
Ymask = (Ymask > miss);
Ymiss = Ymask.*Y;
Imiss = col2im(Ymiss,[p p],size(I),'distinct');

save(fname, 'Y', 'Ymask','Ymiss','Imiss');

%% Dictionary Learning
D = D0;
for iter = 1:K
    X = omp_mask(Ymiss,D,s,Ymask);
    [D, X] = aksvd_mask(Ymiss,D,X,iter,Ymask);
    errs(iter) = norm(Ymask.*(Ymiss - D*X),'fro') / sqrt(sum(sum(Ymask)));
end
save(fname,'D','X','errs','-append');

%Completati codul Matlab
Yc = Ymiss;
Psi = D * X;
Yc(~Ymask) = Psi(~Ymask);
Yc = Yc + repmat(Ymean, size(Y, 1), 1);
Ic = col2im(Yc, [p p], size(I), 'distinct');

imwrite(uint8(I), 'inpaint_I.png');
imwrite(uint8(Imiss), 'inpaint_Imiss.png');
imwrite(uint8(Ic), 'inpaint_Ic.png');

% Inpainted vs. Original
ipsnr = psnr(Ic, I, 255);
issim = ssim(Ic, I, 'DynamicRange', 255);
sprintf('psnr=%f ssim=%f\n', ipsnr, issim);

%% Show images
subplot(1,3,1);imshow(I,[0,255]);title('Original');
subplot(1,3,2);imshow(Imiss,[0 255]);title('Missing');
subplot(1,3,3);imshow(Ic,[0 255]);title('Inpainted');
