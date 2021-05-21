function [y_fft, cepstrum, y_cepstrum_low, y_cepstrum_high] = soundAnalysis(soundFile, play_sound)

[ys,fss] = audioread(soundFile);
if play_sound
soundsc(ys,fss);
pause(2)
end
% y = resample(ys,fs,fss);
% y = [y; zeros(N-length(y)+1,1)];

% analise do sinal natural
y_fft = fft(ys,fss);
y_fft = 20*log10(abs(y_fft));
% y_fft = y_fft(1:length(y_fft)/2);

cepstrum = ifft(log(abs(y_fft)));

Ycep_low = cepstrum;
Ycep_low(41:(end-40+1))=0;
Ycep_high = cepstrum;
Ycep_high([1:40 (end-40+2):end])=0;

y_cepstrum_low = real(fft(Ycep_low));
y_cepstrum_high = real(fft(Ycep_high));

end