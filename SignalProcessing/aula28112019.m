%
% Analise Espectral: sinais reais
%

% Ruido Branco
N = 4e4;
% Gaussiano
x_gauss = randn(N,1);
aux = rand(N,1);
x_unif = (aux - mean(aux))/std(aux)*std(x_gauss);

subplot(3,2,1)
hist(x_gauss,(-1:2/sqrt(N):1)*4);
axis([-3 3 0 4*sqrt(N)]);
subplot(3,2,2)
hist(x_unif,(-1:2/sqrt(N):1)*4);
axis([-3 3 0 4*sqrt(N)]);
subplot(3,2,3)
X_gauss = 10*log10((abs(fft(x_gauss))).^2/N);
plot(X_gauss)
axis([0 N -30 20])
subplot(3,2,4)
X_unif = 10*log10((abs(fft(x_unif))).^2/N);
plot(X_unif)
axis([0 N -30 20])
subplot(3,2,5)
plot(x_gauss)
subplot(3,2,6)
plot(x_unif)
