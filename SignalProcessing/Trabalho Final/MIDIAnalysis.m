clc
play_sound = true;

oboe_set = ["oboe_A4_15_forte_normal.mp3",...
    "oboe_A4_15_fortissimo_normal.mp3",...
    "oboe_A4_15_mezzo-forte_normal.mp3",...
    "Oboe_MIDI_A4_2.wav",...
   "Oboe_MIDI_A4_2.wav"];

sound_used = oboe_set;

for i = 1:size(sound_used,2)
    soundFile = char(sound_used(i));
    [ys,fss] = audioread(soundFile);
    figure(4) 
    hold on
    if contains(sound_used(i), "MIDI")  
        plot(ys,'--')
    else           
        plot(ys)
    end
    title('Sinais no dominio do tempo')
    xlabel('Amostra')
    ylabel('Amplitude')
    
    [y_fft,~, Ylow, Yhigh] = soundAnalysis(soundFile, play_sound);
%     subplot(4,1,1)
    figure(1)
    if contains(sound_used(i), "MIDI")  
        plot(y_fft,'--')
    else           
        plot(y_fft)
    end
    hold on
    title('resposta em frequencia')
    xlabel('Db')
    ylabel('Hz')
%     subplot(3,1,2)
    figure(2)
    if contains(sound_used(i), "MIDI")  
        plot(Ylow,'--')
    else           
        plot(Ylow)
    end
    hold on
    title('analise Cepstral - resposta do sistema') 
%     subplot(3,1,3)
    figure(3)
    if contains(sound_used(i), "MIDI")  
        plot(Yhigh,'--')
    else           
        plot(Yhigh)
    end
    hold on
    title('analise Cepstral - Excitação') 
end
figure(1)
legend(sound_used)
figure(2)
legend(sound_used)
figure(3)
legend(sound_used)
figure(4)
legend(sound_used)
