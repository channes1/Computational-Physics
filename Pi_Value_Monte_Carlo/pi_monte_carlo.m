clc;
clear all;
n=input('Number of points: ');
x=rand(n,1);
y=rand(n,1);
figure('color','white');
hold on

axis square;
x1=x-0.5;
y1=y-0.5; %Center of circle at (0.5,0.5)
r=x1.^2+y1.^2;
u=0;   %Number of points inside the circle
for i=1:n
    if r(i)<=0.25
        u=u+1;
        plot(x(i),y(i),'k.');
    else

        plot(x(i),y(i),'b.');
    end
end
val=u/(0.25*n);
disp(val);
hold off

%Output
% Number of points: 10000
  % 3.1448