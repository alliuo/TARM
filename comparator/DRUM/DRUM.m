%% nxn approximate multiplier DRUM
%% k is the bit width of the accurate multiplier
function M = DRUM(x,y,n,k)

x = double(uint64(x));
y = double(uint64(y));
len = length(x);

lx = zeros(1,len);
ly = zeros(1,len);
lx(x~=0) = log2(x(x~=0));
ly(y~=0) = log2(y(y~=0));
lxz = floor(lx);
lyz = floor(ly);
lxz(lxz<=k-1) = k-1;
lyz(lyz<=k-1) = k-1;

xt = x;
yt = y;
xt(lxz>=k) = floor((floor(xt(lxz>=k)./2.^(lxz(lxz>=k)-k+2))+2^(-1)).*2.^(lxz(lxz>=k)-k+2));
yt(lyz>=k) = floor((floor(yt(lyz>=k)./2.^(lyz(lyz>=k)-k+2))+2^(-1)).*2.^(lyz(lyz>=k)-k+2));

M = xt.*yt;
