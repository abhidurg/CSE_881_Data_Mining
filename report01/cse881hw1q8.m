A = load('spambase.data');
N = size(A,1);
seed = 10;     
rng(seed);    
A = A(randperm(N),:);

X = A(:,1:end-1); %split predictors into X
y = A(:,end);   %split class label into y

NumfoldsOuter = 10;      %outerloop
NumfoldsInner = 5;  %innerloop
Lambda = logspace(-4,3,11);     % hyperparameters for linear SVM
bc = logspace(-3,3,11);         % hyperparameters for nonlinear SVM
ks = logspace(-3,3,11);         % hyperparameters for nonlinear SVM

indexes = crossvalind('KFold',N,NumfoldsOuter);  % indices for cross validation

Linear_Predictions = zeros(N,1);
Nonlinear_Predictions = zeros(N,1);

for fold=1:NumfoldsOuter
    guide_msg = sprintf('Executing fold %d', fold);
    disp(guide_msg);
    testIdx = find(indexes == fold);
    trainIdx = find(indexes ~= fold);
    disp(' ---- Executing linear SVM ---');
    linearSVMm = fitclinear(X(trainIdx,:),y(trainIdx),'Kfold',NumfoldsInner,'Learner','svm','Lambda',Lambda);
    ce = kfoldLoss(linearSVMm);
    [~,bestIdx] = min(ce);
    bestLambda = Lambda(bestIdx(1));
    linearSVMm = fitclinear(X(trainIdx,:),y(trainIdx),'Learner','svm','Lambda',bestLambda);
    Linear_Predictions(testIdx) = predict(linearSVMm, X(testIdx,:));
    cvloss = zeros(length(bc),length(ks));

    disp('  --- Executing nonlinear SVM ---');
    for i=1:11
        for j=1:11
            nonlinearSVMm = fitcsvm(X(trainIdx,:),y(trainIdx),'Standardize',true,'KernelFunction','RBF','KernelScale',ks(j),'BoxConstraint',bc(i),'Kfold',NumfoldsInner);
            cvloss(i,j) = kfoldLoss(nonlinearSVMm);
        end
    end

    [bcIdx,ksIdx] = find(cvloss == min(cvloss(:)));
    bestbc = bc(bcIdx(1));
    bestks = ks(ksIdx(1));
    nonlinearSVMm = fitcsvm(X(trainIdx,:),y(trainIdx),'Standardize',true,'KernelFunction','RBF','KernelScale',bestks,'BoxConstraint',bestbc);
    Nonlinear_Predictions(testIdx) = predict(nonlinearSVMm, X(testIdx,:));
    
end

cp1 = classperf(y);
classperf(cp1,Linear_Predictions);
cp1.DiagnosticTable
cp1.ErrorRate

cp2 = classperf(y);
classperf(cp2,Nonlinear_Predictions);
cp2.DiagnosticTable
cp2.ErrorRate

save q8answers.mat
