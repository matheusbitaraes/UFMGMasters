%
% Processamento de sinais 24/10/2019
%

% Sintese de um segundo de uma senoide de 350 Hz (Fa natural 3)
fs = 16000; % Amostras por segundo
T = 1; % Tempo
N = fs*T; % Numero de amostras
f1 = 350; % Frequencia fundamental
t = 0:1/fs:T;
x = sin(2*pi*f1*t);
soundsc(x,fs);

% Fa natural 3 tocado em um trompete
[ys,fss] = audioread('Fá_Natural_3.wav');
soundsc(ys,fss);
y = resample(ys,fs,fss);
y = [y; zeros(N-length(y)+1,1)];
Y = fft(y,fs);
plot(20*log10(abs(Y)))
aux = abs(Y);
aux(find(aux<10))=0;
aux = filter([1 2 3 4 5 4 3 2 1]/25,1,aux);
[pks,locs] = findpeaks(aux);
f0 = mode(diff(locs));
K = 22;
xk = 0;
figure(1)
for k = 1:K
  xk=xk-real(Y(k*f0))*cos(2*pi*k*f0*t)+imag(Y(k*f0))*sin(2*pi*k*f0*t);
  soundsc(xk,fs);
  plot(t,y/std(y),'b',t,xk/std(xk),'r');
  axis([0.3 0.32 -4 4])
%  pause
end

% Analise CEPSTRAL

Ycep = ifft(log(abs(Y)));
figure(2)
plot(Ycep)
axis([-2.5 1.5 0 500])

Ycep_low = Ycep;
Ycep_low(41:(end-40+1))=0;
Ycep_high = Ycep;
Ycep_high([1:40 (end-40+2):end])=0;

Ylow = real(fft(Ycep_low));
Yhigh = real(fft(Ycep_high));
figure(3)
subplot(2,1,1)
plot(Ylow)
subplot(2,1,2)
plot(Yhigh)
