% Monte Carlo simulation of particle diffusion

clear all;
clc;
clf;
a1=fopen('particle.dat','w'); % Store data in file
t=0:50000;
nop=10000;  % Total number of particles
x=[nop];
nl=nop;   % Asssumed that all particles are on left side of the box at t=0
for i=1:50000
    if rand()<(nl/nop)
        nl=nl-1;
    else

        nl=nl+1;
    end
    fprintf(a1,'%d\t%d\n',t(i),nl);
    x=[x;nl];
end
y=(nop/2)*(1+exp(-2*t/nop)); %Analytic solution
plot(t,x,'b',t,y,'r')
legend('Monte-Carlo','Analytic')
xlabel('Time in sec')
ylabel('No. of particles')
fclose(a1); %Close file