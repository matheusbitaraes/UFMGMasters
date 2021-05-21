%
% Processamento de sinais
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
soundsc(y,fss);
y = resample(ys,fs,fss);
Y = fft(y,fs);
plot(20*log10(abs(Y)))
aux = abs(Y);
aux(find(aux<10))=0;
aux = filter([1 2 3 4 5 4 3 2 1]/25,1,aux);
[pks,locs] = findpeaks(aux);

