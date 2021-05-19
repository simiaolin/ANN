%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
n=50;
iter = 100;
for i=1:n
    a={rands(3,1)};                         % generate an initial point                   
    a1={[a{1,1}(1,1); -a{1,1}(2,1); a{1,1}(3,1)]};
    a2={[a{1,1}(1,1); a{1,1}(2,1); -a{1,1}(3,1)]};
    a3={[-a{1,1}(1,1); a{1,1}(2,1); a{1,1}(3,1)]};

    a4={[-a{1,1}(1,1); -a{1,1}(2,1); a{1,1}(3,1)]};
    a5={[-a{1,1}(1,1); a{1,1}(2,1); -a{1,1}(3,1)]};
    a6={[a{1,1}(1,1); -a{1,1}(2,1); -a{1,1}(3,1)]};
    a7={[-a{1,1}(1,1); -a{1,1}(2,1); -a{1,1}(3,1)]};
    [y,Pf,Af] = sim(net,{1 iter},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    p1 = plot3(start(1,1),start(2,1),start(3,1),'bx');  % plot evolution
    hold on;
    p2 = plot3(record(1,:),record(2,:),record(3,:),'r');
    p3 = plot3(record(1,50),record(2,50),record(3,50),'gO');  % plot the final point with a green circle
   plotother(a1);
   plotother(a2);
   plotother(a3);
   plotother(a4);
   plotother(a5);
   plotother(a6);
   plotother(a7);
end
zero = {[0;0; 0]};
plotother(zero);
grid on;
legend([p1 p2 p3],{'initial state','time evolution','attractor'},'Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');
