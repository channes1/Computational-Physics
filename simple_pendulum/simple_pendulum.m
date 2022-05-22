clc;
clear all;

L = 1;%Length
ang = 1; %Angle
ti = 0;%Initial time
tf= 30; %Final time
h=0.004;%Step size
g=9.8; %Acceleration due to gravity

w=0;
index=1;%index variable

while(ti<=tf)
    f=-g*sin(ang)/L;
    
    ta(index)=ti;
    tha(index)=ang;
    wa(index)=w;
    fa(index)=f;
    ang=ang+w*h;
    w=w+f*h;
    ti=ti+h;
    index=index+1;
end

t_b=[ta' tha' wa' fa'];
plot(ta,tha,'b');
xlabel('time');
ylabel('angle');
title('Simple pendulum');
