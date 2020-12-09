function [pred] = applyKernel(Xtrain, Xtest, alpha, gamma)
%
% function [pred] = applyKernel(Xtrain, Xtest, alpha, gamma)
%
% This function will apply kernel regression to the given test
% data. You need to provide the following input: 
% 1. Xtrain: a matrix of predictor variables for training data
% 2. Xtest: a matrix of predictor variables for test data.
% 3. alpha: output of kernel regression (see lecture notes).
% 4. gamma: kernel width parameter.
%
% The function will return the predicted values for the test data
% in a vector called pred.

blocksize = 5000;

Ntrain = size(Xtrain,1);    % number of training examples
Ntest = size(Xtest,1);      % number of test examples
numBlocks = ceil(Ntest/blocksize);

pred = zeros(Ntest,1);

for i=1:numBlocks
    startIdx = (i-1)*blocksize + 1;
    if i < numBlocks
        endIdx = i*blocksize;
    else
        endIdx = Ntest;
    end;
    
    X = [Xtrain; Xtest(startIdx:endIdx,:)];
    dist = pdist(X);	  
    dist = squareform(dist);
    dist = dist(1:Ntrain,Ntrain+1:end);
    K = exp(-gamma * dist.^2);
    pred(startIdx:endIdx) = (alpha'*K)';
end;

