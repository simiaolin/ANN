%% Time Series Forecasting Using Deep Learning
% This example shows how to forecast time series data using a long short-term 
% memory (LSTM) network.
% 
% To forecast the values of future time steps of a sequence, you can train a 
% sequence-to-sequence regression LSTM network, where the responses are the training 
% sequences with values shifted by one time step. That is, at each time step of 
% the input sequence, the LSTM network learns to predict the value of the next 
% time step.
% 
% To forecast the values of multiple time steps in the future, use the |predictAndUpdateState| 
% function to predict time steps one at a time and update the network state at 
% each prediction.
% 
% This example uses the data set |chickenpox_dataset|. The example trains an 
% LSTM network to forecast the number of chickenpox cases given the number of 
% cases in previous months.
%% Load Sequence Data
% Load the example data. |chickenpox_dataset| contains a single time series, 
% with time steps corresponding to months and values corresponding to the number 
% of cases. The output is a cell array, where each element is a single time step. 
% Reshape the data to be a row vector.

clear
clc

load lasertrain.dat
load laserpred.dat


%%

%% 
% Partition the training and test data. Train on the first 90% of the sequence 
% and test on the last 10%.

numTimeStepsTrain = floor(numel(lasertrain));

dataTrain = lasertrain';
dataTest = laserpred';

data = [dataTrain dataTest];
%% Standardize Data
% For a better fit and to prevent the training from diverging, standardize the 
% training data to have zero mean and unit variance. At prediction time, you must 
% standardize the test data using the same parameters as the training data.

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;
%% Prepare Predictors and Responses
% To forecast the values of future time steps of a sequence, specify the responses 
% to be the training sequences with values shifted by one time step. That is, 
% at each time step of the input sequence, the LSTM network learns to predict 
% the value of the next time step. The predictors are the training sequences without 
% the final time step.

% numbr of lags
p = 50;

[XTrain,YTrain]=getTimeSeriesTrainData(dataTrainStandardized', p); 


numFeatures = p;
numResponses = 1;
numHiddenUnits = 100;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(40)
    lstmLayer(40)
    fullyConnectedLayer(numResponses)
    regressionLayer];
max_epoch = 20011.84111
options = trainingOptions('adam', ...
    'MaxEpochs',max_epoch, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',max_epoch/2, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Train LSTM Network
% Train the LSTM network with the specified training options by using |trainNetwork|.

net = trainNetwork(XTrain,YTrain,layers,options);
net = predictAndUpdateState(net,XTrain);

YPred = zeros(size(dataTest));
% create and initialize the temp_input vector with last # lag train dataset
temp_input = dataTrainStandardized(end-p+1:end);


numTimeStepsTest = size(dataTest,2);
for i = 1:numTimeStepsTest
    [net,temp_output] = predictAndUpdateState(net,temp_input','ExecutionEnvironment','cpu');

    YPred(i) = temp_output;

    % update temp_input

    for j = 1:(p-1)
        temp_input(j) = temp_input(j+1);
    end

    temp_input(p) = temp_output;
    
end
%% 
% Unstandardize the predictions using the parameters calculated earlier.

YPred = sig*YPred + mu;
%% 
% The training progress plot reports the root-mean-square error (RMSE) calculated 
% from the standardized data. Calculate the RMSE from the unstandardized predictions.

YTest = dataTest;
rmse = sqrt(mean((YPred-YTest).^2))
disp(rmse)
%% 
% Plot the training time series with the forecasted values.

figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+size(dataTest,2));
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])
%% 
% Compare the forecasted values with the test data.

figure
subplot(3,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(3,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)


subplot(3,1,3)
postregm(YPred,YTest); % perform a linear regression analysis and plot the result

