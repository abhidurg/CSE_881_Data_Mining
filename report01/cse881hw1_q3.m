%part a)
D = importdata('OnlineNewsPopularity.csv');
predictor_variables = D.data(:, 1:58); %predictor
response_variable = log(D.data(:, end)); %log of response variable, last column
r = rank(predictor_variables); %56
piv_tol = max(size(predictor_variables)*eps*norm(predictor_variables)); %adujusting pivot tolerance
[R, p] = rref(predictor_variables, piv_tol); %p shows that columns 37 and 42 are dependent
discarded = predictor_variables(:, p); %discards the depended columns


%part b)
training_dataX = discarded(1:2000,:);
testing_dataX = discarded(2001:end, :);

%standardize each perdictor variable (trainingX)
Standdev = std(training_dataX);
Aver = mean(training_dataX);
[R,C] = size(training_dataX);
for c = 1:C
    for r = 1:R
        training_dataX(r,c) = (training_dataX(r, c)-Aver(c))/Standdev(c);
    end
end



training_dataY = response_variable(1:2000,:);
testing_dataY = response_variable(2001:end, :);

training_dataX = [training_dataX ones(size(training_dataX,1), 1)]; %add 1s predictor variables
testing_dataX = [testing_dataX ones(size(testing_dataX,1), 1)];


%MLR
mlr_reg = regress(training_dataY, training_dataX);
[B, I] = maxk(abs(mlr_reg), 10);
B_mlr = mlr_reg(I);
disp(B_mlr);
disp(I);

