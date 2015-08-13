% tests random kitchen sinks
% clear all; clc; close all;
function test_rks()

path(path(), genpath('~/Dropbox/Matlab/utils')); % sq_dist from here
path(path(), genpath('~/Dropbox/Matlab/gpml-matlab-v3.6-2015-07-07'));
rng(10101,'twister');

%% General settings
N       = 20;
d       = 1; % original dimensionality of input space
D       = 1000; % dimensionality of output space
covfunc = 'covSEiso';
ell     = 1/2; 
sf      = 1; 
hyp.cov = log([ell; sf]);
sigma2y = 1e-3; 
sigma2w = 1;

%% Generates data
[x, xstar, y, K, cholK] =   getData(N, d, covfunc, hyp, sigma2y);

%% Generates random Features 
Z       = randn(D,d);
sigma_z = getOptimalSigmaz(ell);
PHI     = getRandomRBF(Z, sigma_z, x);

%% Approximate kernel
analyzeKernels(K, PHI);


%% prediction  at xstar with GP model
[mPredGP, varPredGP] =  predictGP(covfunc, hyp, x, xstar, cholK, y);

%% prediction with rks
[mPredRKS, varPredRKS]  = predictRKS(PHI, Z, sigma_z, sigma2w, sigma2y, xstar, y);

%% Plots
figure;
plot_confidence_interval(xstar,mPredRKS,sqrt(varPredRKS), [], 1, 'b', [0.7 0.9 0.95]); 
hold on;
plot_confidence_interval(xstar,mPredGP,sqrt(varPredGP), [], 0, 'r', 'r'); hold on;
plot(x, y, 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k'); hold on; % data
xlabel('x');
ylabel('y');
%saveas(gcf, 'alir.eps', 'epsc');
saveas(gcf, 'me.eps', 'epsc');
%
% 
% Predictive distribution by GP


return;

%% 
function [mPredRKS, varPredRKS]  = predictRKS(PHI, Z, sigma_z, sigma2w, sigma2y, xstar, y)
% learning
D          = size(Z,1);
Sigmayw    = (sigma2y/sigma2w)*eye(2*D,2*D);
W          = (PHI'*PHI +  Sigmayw) \(PHI'*y);
% prediction
PHIstar    = getRandomRBF(Z, sigma_z, xstar);
mPredRKS   =  PHIstar*W;
L          = chol(PHI'*PHI + Sigmayw, 'lower');
v          = L\PHIstar';
varPredRKS =  sigma2y*sum(v.*v,1)';
return;


%% [mPredGP, varPredGP] =  predictGP(covfunc, hyp, x, xstar, cholK, y)
function [mPredGP, varPredGP] =  predictGP(covfunc, hyp, x, xstar, cholK, y)
kstar   = feval(covfunc, hyp.cov, x, xstar);
mPredGP = kstar'*(solve_chol(cholK',y)); 
% now the variances
kss       = feval(covfunc, hyp.cov, xstar, 'diag'); % only diagonal requested
v         = cholK\kstar;
varPredGP = kss - sum(v.*v, 1)'; 
return;


%%  function getData()
function [x, xstar, y, K, L] =  getData(N, d, covfunc, hyp, sigma2y)
%% Generates input
x            = -2 + 4*rand(N, d);
if ( d == 1 )
    x = sort(x);
end
xstar = linspace(-2, 2, 100)';
K = feval(covfunc, hyp.cov, x);
L  = chol(K + sigma2y*eye(N,N), 'lower');
Z  = randn(N,1);
f  = L*Z;
y  = f;


return;


%% function analyzeKernels(K, PHI)
function analyzeKernels(K, PHI)
Ktilde  = PHI*PHI';
Diff    = K-Ktilde;
subplot(1,3,1); imagesc(K); colormap(gray); colorbar; title('True');
subplot(1,3,2); imagesc(Ktilde); colormap(gray); colorbar; title('Approx.');
subplot(1,3,3); imagesc(Diff); colormap(gray); colorbar; title('Diff.');
return;
















