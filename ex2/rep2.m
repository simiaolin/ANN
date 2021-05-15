%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
n=20;
iteration = 49;
for i=1:n
    a={rands(2,1)};                     % generate an initial point 
    b= {[a{1,1}(1,1); -a{1,1}(2,1)]}; 
   [y,Pf,Af] = sim(net,{1 iteration},{},a);   % simulation of the network for 50 timesteps              
   [y2, pf, af ] = sim(net, {1 iteration}, {}, b);
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    record_b = [cell2mat(b) cell2mat(y2)];
    start_b = cell2mat(b);
    p1 = plot(start(1,1),start(2,1),'bx');
    hold on;
    p2 = plot(record(1,:),record(2,:),'r'); % plot evolution
   
    
    p3 = plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
    
    p4 = plot(start_b(1,1), start_b(2,1), 'bx', record_b(1,:), record_b(2,:), 'r');
    p5 = plot(record_b(1, 50), record_b(2, 50), 'gO');
    
end
legend([p1 p2 p3] , {'initial state','time evolution','attractor'}, 'Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
