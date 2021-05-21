%
% Analise Espectral
%
n = 1:320;
x = cos(2*pi*(n-1/2)/4);
k0 = -80;
for k = -80:80
  aux = 0;
  for n = 81:240
    aux = aux + (x(n).*x(n-k));
  end
  R(k-k0+1) = aux/160;
end
k = -80:80;
subplot(2,1,1)
stem(k,R,'r')

w = (0:0.01:1)'*2*pi;
S = sum(R.*exp(-j*w*k),2);
subplot(2,1,2)
plot(w,S)


