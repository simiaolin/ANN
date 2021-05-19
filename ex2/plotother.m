
function [] = plotother(a)
    T = [1 1 1; -1 -1 1; 1 -1 -1]';
    net = newhop(T);
    n=10;
    iter = 50;
    [y,Pf,Af] = sim(net,{1 iter},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results
    plot3(start(1,1),start(2,1),start(3,1),'bx');  % plot evolution
    hold on;
    plot3(record(1,:),record(2,:),record(3,:),'r');
    plot3(record(1,50),record(2,50),record(3,50),'gO');  % plot the final point with a green circle

