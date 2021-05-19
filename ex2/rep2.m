%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
n=50;
iteration = 49;
for i=1:n
    a={rands(2,1) * 1.5};                     % generate an initial point 
    b= {[a{1,1}(1,1); -a{1,1}(2,1)]}; 
    b1= {-[a{1,1}(1,1); a{1,1}(2,1)]};
    b2= {-[a{1,1}(1,1); -a{1,1}(2,1)]};

    [y,Pf,Af] = sim(net,{1 iteration},{},a);   % simulation of the network for 50 timesteps

    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results
    p1 = plot(start(1,1),start(2,1),'bx');
    hold on;
    p2 = plot(record(1,:),record(2,:),'r'); % plot evolution
    p3 = plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle

    [y2, pf, af] = sim(net, {1 iteration}, {}, b);
    record_b = [cell2mat(b) cell2mat(y2)];
    start_b = cell2mat(b);
    p4 = plot(start_b(1,1), start_b(2,1), 'bx', record_b(1,:), record_b(2,:), 'r');
    p5 = plot(record_b(1, 50), record_b(2, 50), 'gO');

    [y21, pf1, af1] = sim(net, {1 iteration}, {}, b1);
    record_b1 = [cell2mat(b1) cell2mat(y21)];
    start_b1 = cell2mat(b1);
    plot(start_b1(1,1), start_b1(2,1), 'bx', record_b1(1,:), record_b1(2,:), 'r');
    plot(record_b1(1, 50), record_b1(2, 50), 'gO');


    [y22, pf2, af2] = sim(net, {1 iteration}, {}, b2);
    record_b2 = [cell2mat(b2) cell2mat(y22)];
    start_b2 = cell2mat(b2);
    plot(start_b2(1,1), start_b2(2,1), 'bx', record_b2(1,:), record_b2(2,:), 'r');
    plot(record_b2(1, 50), record_b2(2, 50), 'gO');


end

zero = {[0;0]};
[yz,Pfz,Afz] = sim(net,{1 iteration},{},zero);
recordz=[cell2mat(zero) cell2mat(yz)];
start=cell2mat(zero);                  % formatting results 
plot(start(1,1),start(2,1),'bx');
plot(recordz(1,:),recordz(2,:),'r');
plot(recordz(1,50),recordz(2,50),'gO');
legend([p1 p2 p3] , {'initial state','time evolution','attractor'}, 'Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
