
% build a new target Tnew
% consider the largest 5 digits of your student number in descending order, represented
% by d1; d2; d3; d4; d5 (with d1 the largest digit).
% student number: 0481422
d1 = 9;d2 = 8;d3 = 5;d4 = 2;d5 = 2;

% Tnew = (d1T1 + d2T2 + d3T3 + d4T4 + d5T5)/(d1 + d2 + d3 + d4 + d5).
load('Data_Problem1_regression.mat');
Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);


% training set
T_index = randperm(length(X1),1000);
X1_train = X1(T_index);
X2_train = X2(T_index);
X_train = [X1_train, X2_train];
T_train = Tnew(T_index);
x_train = X_train.';
t_train = T_train.';

% Plot the surface of your training set using the Matlab function scatteredInterpolant,plot3 and mesh.
F = scatteredInterpolant(X1_train,X2_train,T_train);

xlin = linspace(min(X1_train),max(X1_train),100);
ylin = linspace(min(X2_train),max(X2_train),100);

f1 = figure('Name','Surface of training set')
[X1_M,X2_M] = meshgrid(xlin,ylin);
T_M = F(X1_M,X2_M);
mesh(X1_M,X2_M,T_M);
title('Surface of training set')
xlabel('X1'), ylabel('X2'), zlabel('Target')
legend('Sample data','Interpolated query data','Location','Best')

% validation set
V_index = randperm(length(X1),1000);
X1_val = X1(V_index);
X2_val = X2(V_index);
X_val = [X1_val, X2_val]; 
T_val = Tnew(V_index);
x_val = X_val.';
t_val = T_val.';


% test set 
Test_index = randperm(length(X1),1000);
X1_test = X1(Test_index);
X2_test = X2(Test_index);
X_test = [X1_test, X2_test]; 
T_test = Tnew(Test_index);
x_test = X_test.';
t_test = T_test.';

% traingd gradient descent
% traingda gradient descent with adaptive learning rate
% traincgf Fletcher-Reeves conjugate gradient algorithm
% traincgp Polak-Ribiere conjugate gradient algorithm
% trainbfg - BFGS (quasi Newton)
% trainlm - Levenberg - Marquardt

% Configuration:
alg = 'trainbfg';% First training algorithm to use
H = 50;% Number of neurons in the hidden layer
delta_epochs = [1,14,85];% Number of epochs to train in each step
epochs = cumsum(delta_epochs);


%creation of networks
net=feedforwardnet(H,alg);
net=configure(net,x_train,t_train);% Set the input and output sizes of the net
net.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
net=init(net);% Initialize the weights (randomly)


%training 
net.trainParam.epochs=delta_epochs(1);  % set the number of epochs for the training 
net=train(net,x_train,t_train);   % train the networks

t1=sim(net,x_train);
v1=sim(net,x_val);
a1=sim(net,x_test);% simulate the networks with the input vector X_train
A1 = a1.';

net.trainParam.epochs=delta_epochs(2);  % set the number of epochs for the training 
net=train(net,x_train,t_train);   % train the networks

t2=sim(net,x_train);
v2=sim(net,x_val);
a2=sim(net,x_test);% simulate the networks with the input vector X_train
A2 = a2.';

net.trainParam.epochs=delta_epochs(3);  % set the number of epochs for the training 
net=train(net,x_train,t_train);   % train the networks

t3=sim(net,x_train);
v3=sim(net,x_val);
a3=sim(net,x_test);% simulate the networks with the input vector X_train
A3 = a3.';



%validation 

Er = t1 - t_val;
Se = Er.^2;
MSE_t1 = mean(Se);

Er = t2 - t_val;
Se = Er.^2;
MSE_t2 = mean(Se);

Er = t2 - t_val;
Se = Er.^2;
MSE_t3 = mean(Se);

MSE_Train = [MSE_t1 MSE_t2 MSE_t3] 

Er = v1 - t_val;
Se = Er.^2;
MSE_v1 = mean(Se);

Er = v2 - t_val;
Se = Er.^2;
MSE_v2 = mean(Se);

Er = v2 - t_val;
Se = Er.^2;
MSE_v3 = mean(Se);

MSE_Val = [MSE_v1 MSE_v2 MSE_v3]

figure('Name','Iteration VS. Error')

plot(epochs, MSE_Train, epochs, MSE_Val)
legend('training set','validation set');
title('Iteration VS. Error');
xlabel('Iteration step');
ylabel('E');


%evaluation
F2 = scatteredInterpolant(X1_test,X2_test,T_test);
FA1 = scatteredInterpolant(X1_test,X2_test,A1);
FA2 = scatteredInterpolant(X1_test,X2_test,A2);
FA3 = scatteredInterpolant(X1_test,X2_test,A3);

xlin = linspace(min(X1_test),max(X1_test),100);
ylin = linspace(min(X2_test),max(X2_test),100);
[X1_M,X2_M] = meshgrid(xlin,ylin);

% f2 = figure('Name','Surface of test set')
figure('Name',alg)
subplot(3,3,1);
T_M = F2(X1_M,X2_M);
T_A = FA1(X1_M,X2_M);
s = mesh(X1_M,X2_M,T_M);
s.EdgeColor = [0 1 0];
title([num2str(epochs(1)),' epochs - Surface of test set and network approximation']);

xlabel('X1'), ylabel('X2'), zlabel('Target');
hold on
s = mesh(X1_M,X2_M,T_A);
s.EdgeColor = [1 0 0];
legend('target','Test Set','Approximation','Location','north');

subplot(3,3,2);
mesh(X1_M,X2_M,T_M - T_A);
Er = a1 - t_test;
Se = Er.^2;
MSE = mean(Se);
title(['MSE: ',num2str(MSE),' Surface of error level curves']);
xlabel('X1'), ylabel('X2'), zlabel('Target');

subplot(3,3,3)
postregm(a1,t_test);

subplot(3,3,4);
T_M = F2(X1_M,X2_M);
T_A = FA2(X1_M,X2_M);
s = mesh(X1_M,X2_M,T_M);
s.EdgeColor = [0 1 0];
title([num2str(epochs(2)),' epochs - Surface of test set and network approximation']);

xlabel('X1'), ylabel('X2'), zlabel('Target');
hold on
s = mesh(X1_M,X2_M,T_A);
s.EdgeColor = [1 0 0];
legend('target','Test Set','Approximation','Location','north');

subplot(3,3,5);
mesh(X1_M,X2_M,T_M - T_A);
Er = a2 - t_test;
Se = Er.^2;
MSE = mean(Se);
title(['MSE: ',num2str(MSE),' Surface of error level curves']);
xlabel('X1'), ylabel('X2'), zlabel('Target');

subplot(3,3,6)
postregm(a2,t_test);

subplot(3,3,7);
T_M = F2(X1_M,X2_M);
T_A = FA3(X1_M,X2_M);
s = mesh(X1_M,X2_M,T_M);
s.EdgeColor = [0 1 0];
title([num2str(epochs(3)),' epochs - Surface of test set and network approximation']);

xlabel('X1'), ylabel('X2'), zlabel('Target');
hold on
s = mesh(X1_M,X2_M,T_A);
s.EdgeColor = [1 0 0];
legend('target','Test Set','Approximation','Location','north');

subplot(3,3,8);
mesh(X1_M,X2_M,T_M - T_A);
Er = a3 - t_test;
Se = Er.^2;
MSE = mean(Se);
title(['MSE: ',num2str(MSE),' Surface of error level curves']);
xlabel('X1'), ylabel('X2'), zlabel('Target');

subplot(3,3,9)
postregm(a3,t_test);
