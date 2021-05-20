clear all
close all
nntraintool('close');
nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
rng('default')


% Load the training data into memory
load('digittrain_dataset.mat');

% Layer 1
hiddenSize1 = 400;
hiddenSize2 = 100;
max_epoch1  = 400;
max_epoch2  = 200;
max_epoch3  = 100;

"stack first layer"
tic
autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
    'MaxEpochs',max_epoch1, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false);
toc

% figure;
% plotWeights(autoenc1);
feat1 = encode(autoenc1,xTrainImages);

% Layer 2
"stack second layer"
tic
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',max_epoch2, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false);
toc 
feat2 = encode(autoenc2,feat1);

"stack soft layer"
% Layer 3
tic
softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',max_epoch3);
toc

% Deep Net
deepnet = stack(autoenc1,autoenc2,softnet);


% Test deep net
imageWidth = 28;
imageHeight = 28;
inputSize = imageWidth*imageHeight;
load('digittest_dataset.mat');
xTest = zeros(inputSize,numel(xTestImages));
for i = 1:numel(xTestImages)
    xTest(:,i) = xTestImages{i}(:);
end
y = deepnet(xTest);
% figure;
% plotconfusion(tTest,y);
StackEncodersAcc=100*(1-confusion(tTest,y))


% Test fine-tuned deep net
xTrain = zeros(inputSize,numel(xTrainImages));
for i = 1:numel(xTrainImages)
    xTrain(:,i) = xTrainImages{i}(:);
end
"fine tuned"
tic 
deepnet = train(deepnet,xTrain,tTrain);
toc
y = deepnet(xTest);
% figure;
% plotconfusion(tTest,y);
FinetunedAcc=100*(1-confusion(tTest,y))
% view(deepnet)

%Compare with normal neural network (1 hidden layers)
% net = patternnet(100);
% net=train(net,xTrain,tTrain);
% y=net(xTest);
% plotconfusion(tTest,y);
% classAcc=100*(1-confusion(tTest,y))
% view(net)
% 
% %Compare with normal neural network (2 hidden layers)
% net = patternnet([100 50]);
% net=train(net,xTrain,tTrain);
% y=net(xTest);
% plotconfusion(tTest,y);
% classAcc=100*(1-confusion(tTest,y))
% view(net)