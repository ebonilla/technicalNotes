% tests unscented transform
clear all; clc;
D = 10;
m = randn(D,1);
C = randn(D,D); C = C*C';
L = chol(C, 'lower');
kappa = 1/2;

S = 2*D + 1; % number of samples
X = zeros(D,S); % samples
X(:,1) = m;
X(:,2:2+D-1) =  repmat(m,1,D) + sqrt(D + kappa)*L;
X(:,2+D:S)   = repmat(m,1,D) - sqrt(D + kappa)*L;
