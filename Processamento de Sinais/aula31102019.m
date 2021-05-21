%
% Processos Auto-Regressivos (AR Processes)
%
N = 3e4;
e = [0 0 0 0 1 zeros(1,N/3-2) 1 zeros(1,N/3-2) 1 zeros(1,N/3-3)];
a1 = 1;
a2 = -1 + 0.001;
a3 = sqrt(2);
a4 = -1 + 0.001;
x1 = zeros(size(e));
x2 = zeros(size(e));
for k = 5:N
  x1(k) = a1*x1(k-1) + a2*x1(k-2) + e(k);
  x2(k) = a3*x2(k-1) + a4*x2(k-2) + e(k);
end
y = x1 + x2;
f = (0:(N-1))*8/N;
figure(1)
clf
plot(f,20*log10(abs(fft(y))));
axis([0 4 -80 80])

p = 12;
p = 16;
alpha = lpc(y,p);
z = filter(1,alpha,e);
hold on
plot(f,20*log10(abs(fft(z))),'r');
axis([0 4 -80 80])
hold off

				% Análise Cepstral

Yceps = ifft(log10(abs(fft(y))));
Zceps = ifft(log10(abs(fft(z))));
figure(2)
plot(Yceps)
hold on
plot(Zceps,'r')
hold off

figure(3)
Yceps_high = Yceps;
Yceps_high([1:5e3 (N-5e3+2):N]) = 0;
Yrec_high = fft(Yceps_high);
plot(real(Yrec_high))

figure(4)
Yceps_low = Yceps;
Yceps_low((5e3+1):(N-5e3+1)) = 0;
Yrec_low = fft(Yceps_low);
plot(real(Yrec_low))


