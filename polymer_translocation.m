function polymer_translocation_nanopore()
% POLYMER_TRANSLOCATION_NANOPORE with GIF output
%
% Simulates polymer translocation and saves an animated GIF.

rng(1);

%% ---------------- Parameters -----------------
N        = 80;   b0=1;  kbond=100;
sigma=1; epsilon=1; rc=2^(1/6)*sigma;
gamma=1; kBT=1; dt=5e-4; tmax=2e3;
sampleEvery=50;

wmem=2.0; rpore=1.5; kwall=200; Fdrive=6.0;
Ly=max(6*rpore,6*sigma+2*N^(1/2)*b0); yPBC=true;

x0=-wmem/2-(N-1:-1:0)'*0.9*b0; y0=zeros(N,1); R=[x0,y0];
nEq=3000;

%% ---------------- GIF setup -----------------
makeGIF = true;
gifFile = 'translocation.gif';
gifEvery = 100;  % capture frame interval

%% ---------------- Helpers -------------------
ex=[1,0];
isInPore = @(r) (abs(r(:,2))<=rpore) & (abs(r(:,1))<=wmem/2);
isInSolid= @(r) (abs(r(:,1))<=wmem/2) & (abs(r(:,2))> rpore);
wrapY = @(y) (y - Ly*round(y/Ly));

function [F, U] = forces(R, withDrive)
    if nargin<2, withDrive=true; end
    Nloc=size(R,1); F=zeros(Nloc,2); U=0;
    % Bonds
    dR=R(2:end,:)-R(1:end-1,:);
    if yPBC, dR(:,2)=dR(:,2)-Ly*round(dR(:,2)/Ly); end
    dist=sqrt(sum(dR.^2,2))+1e-12;
    Fbond=kbond*(1-b0./dist).*dR;
    F(1:end-1,:)=F(1:end-1,:)+Fbond;
    F(2:end,:)=F(2:end,:)-Fbond;
    % Excluded volume
    for i=1:Nloc-1
        d=R(i+1:end,:)-R(i,:);
        if yPBC, d(:,2)=d(:,2)-Ly*round(d(:,2)/Ly); end
        rij=sqrt(sum(d.^2,2));
        mask=rij<rc;
        if any(mask)
            d=d(mask,:); rijm=rij(mask)+1e-12;
            sr6=(sigma./rijm).^6;
            fmag=24*epsilon./rijm.*(2*sr6.^2-sr6);
            fij=fmag.*(d./rijm);
            F(i,:)=F(i,:)-sum(fij,1);
            idx=find(mask); F(i+idx,:)=F(i+idx,:)+fij;
        end
    end
    % Membrane
    insideSolid=isInSolid(R);
    if any(insideSolid)
        Rs=R(insideSolid,:);
        px=(wmem/2-abs(Rs(:,1)));
        sgn=-sign(Rs(:,1));
        Fx=kwall*px.*sgn;
        F(insideSolid,1)=F(insideSolid,1)+Fx;
        py=abs(Rs(:,2))-rpore;
        Fy=-kwall*0.1*py.*sign(Rs(:,2));
        F(insideSolid,2)=F(insideSolid,2)+Fy;
    end
    % Driving
    if withDrive
        mask=isInPore(R);
        F(mask,:)=F(mask,:)+Fdrive*ex;
    end
end

function R=stepBD(R,withDrive)
    [F,~]=forces(R,withDrive);
    D=kBT/gamma;
    R=R+(dt/gamma)*F+sqrt(2*D*dt)*randn(size(R));
    if yPBC, R(:,2)=wrapY(R(:,2)); end
end

%% ---------------- Pre-equilibration ----------
for it=1:nEq, R=stepBD(R,false); end

%% ---------------- Production -----------------
nSteps=ceil(tmax/dt); t=0; w2=wmem/2;
rec=1; transCount=zeros(ceil(nSteps/sampleEvery)+2,1);
timeTrace=zeros(size(transCount));
done=false;

fig=figure('Color','w');

for it=1:nSteps
    t=t+dt; R=stepBD(R,true);
    nTrans=sum(R(:,1)>w2);
    if nTrans==N, fprintf('All beads translocated at t=%.3f\n',t); done=true; end
    if mod(it,sampleEvery)==0 || done
        transCount(rec)=nTrans; timeTrace(rec)=t; rec=rec+1;
    end
    % --- GIF frame ---
    if makeGIF && mod(it,gifEvery)==0
        clf; hold on; axis equal;
        patch([-w2 -w2 w2 w2],[-Ly/2 Ly/2 Ly/2 -Ly/2],[0.9 0.9 0.95],'EdgeColor','none');
        patch([-w2 -w2 w2 w2],[-rpore rpore rpore -rpore],[1 1 1],'EdgeColor','none');
        plot(R(:,1),R(:,2),'-k','LineWidth',1);
        plot(R(:,1),R(:,2),'bo','MarkerFaceColor','b');
        xlim([-N*0.3 N*0.3]); ylim([-Ly/2 Ly/2]);
        xlabel('x'); ylabel('y'); title(sprintf('t=%.2f',t));
        drawnow;
        frame=getframe(fig);
        im=frame2im(frame);
        [A,map]=rgb2ind(im,256);
        if it==gifEvery
            imwrite(A,map,gifFile,'gif','LoopCount',Inf,'DelayTime',0.1);
        else
            imwrite(A,map,gifFile,'gif','WriteMode','append','DelayTime',0.1);
        end
    end
    if done, break; end
end

timeTrace=timeTrace(1:rec-1);
transCount=transCount(1:rec-1);
fprintf('Translocation time = %.3f\n',timeTrace(end));

end
