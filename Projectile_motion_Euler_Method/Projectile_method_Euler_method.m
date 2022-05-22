clc;
clear all;
vi = input('Initial velocity (in m/s): ');
a = input('Angle of projection (in degrees): ');
h = input('Time step (in seconds (small values like 0.05)): ');

a=a*pi/180;
g=9.8;  %acceleration due to gravity
x_max=vi^2*sin(2*a)/g;
y_max=vi^2*sin(a)^2/(2*g);
t_d=2*vi*sin(a)/g;           %total time
x1=0;
y1=0;
figure('color','white');
for t=0:h:t_d+h
    x=vi*t*cos(a);          %analytic solution for x
    y=vi*t*sin(a)-g*t^2/2;  %analytic solution for y

    plot(x,y,'r*',x1,y1,'bo')
    hold all
    xlim([0 1.1*x_max]);
    ylim([0 1.1*y_max]);
    title('Projectile Motion');
    xlabel('Distance');
    ylabel('Height');
    getframe;
    x1=x1+h*vi*cos(a);      %Euler's method for x

    y1=y1+h*(vi*sin(a)-g*t);%Euler's method for y
end

%Output
%Initial velocity (in m/s): 20
%Angle of projection (in degrees): 45
%Time step (in seconds): 0.05