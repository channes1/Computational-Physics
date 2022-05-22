% Double Slit Interference and Diffraction

clear all;
clc
theta_max=pi/50;
while(1)
    a=input('Slit width (in micro meter): ');
    a=a*1e-6;
    d=input('Slit seperation (in mm): ');
    d=d*1e-3;
    l=input('Wavelength (in nm): ');
    l=l*1e-9;
    s=input('Distance between slit and screen (in m): ');
    theta=-theta_max:1e-5:theta_max;
    y=s*tan(theta);
    alpha=pi*a*sin(theta)/l;
    beta=pi*d*sin(theta)/l;
    x1=cos(beta).^2;            % Interference term
    x2=(sin(alpha)./alpha).^2;  % Diffraction term
    x=x1.*x2;                   % Combined effect
    plot(y,x,'b',y,x2,'--r');
    title('Double slit diffraction');
    xlabel('Distance');
    ylabel('Intensity');
    hold all;
    ch= input('Press 1 to continue and 0 to exit: ');
if ch == 0
    break;
end
end

%Output
%Slit width (in micro meter): 40
%Slit seperation (in mm): 0.2
%Wavelength (in nm): 600
%Distance between slit and screen (in m): 0.4
%Press 1 to continue and 0 to exit: 