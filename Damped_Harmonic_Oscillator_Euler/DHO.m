clc;
clear all;
k=1;
m=1;
x=0;
ti=0.01; %Initial time
tf=20; %Final time
vi=15;%initial velocity
index=1;

b=0.4; %Sampling coefficient

h=0.01; %step size

while (ti<=tf)
    a=(-k*x-b*vi)/m; %Acceleration
    ta(index) = ti;
    xa(index) = x;
    va(index) = vi;
    aa(index) = a;
    x=x+vi*h; %Find position
    vi=vi+a*h; %Find velocity
    ti=ti+h;
    index=index+1; %Update index variable
end

plot(ta,xa,'b'); %Time vs position graph
hold on
plot(ta,va,'k'); %Time vs velocity graph
hold on
plot(ta, aa,'r'); %Time vs aaceleration
table(:,1) = ta';
table(:,2) = xa';
table(:,3) = va';
table(:,4) = aa';
disp('t x v a');
disp(table);
xlabel('time');
ylabel('position velocity acceleration');
title('Damped Harmonic Oscillator');
legend('x','v','a');
hold off
hold off