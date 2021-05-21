
close all
clear all

% abre arquivo de sinal
oboes = ["oboe_solo_mitchjohnston.mp3"];
input_sound = oboes;

frequence_reference_table = csvread('MIDI_Table.csv');

% vamos criar mensagens MIDI de acordo com o seguinte formato:
% onset (beats) | duration (beats) | MIDI chanel | MIDI pitch | velocity |
% onset (s) | duration (s)

% MIDI pitch: Nota (ex. C4, que seria o numero 60)
% velocity: o quão alta a nota é tocada (0-127)
% onset (s) e duration (s): mesma coisa das duas primeiras colunas porem em
% segundos

subsampling_time = 0.2; % seconds for dividing package
onset_time = 0.0;
previous_pitch_freq = 0;
for i = 1:size(input_sound,2)
    
    % lê o audio
    [ys,fss] = audioread(char(input_sound(i)));
    %soundsc(ys,fss); %toca o audio
    
    % separa o audio em pacotes de um numero determinado amostras
    subsample_size = fss*subsampling_time;
    subsample_begin = 1;
    n_subsamples = floor(length(ys)/subsample_size);   
    nmat = [];
    
    for j = 1:n_subsamples
        subsample = ys(subsample_begin:subsample_size*j);
        
        % para cada pacote, gera uma nota midi (se for a mesma da anterior,
        % apenas aumenta o tamanho da nota)
        
        % pitch:
        % faz a fft do subset
        Y = fft(subsample,fss);
        y_freq = 20*log10(abs(Y));
        y_freq = y_freq(1:length(y_freq)/2); % pega apenas a primeira metade do sinal pela simetria

        [peakamps, peaklocs] = findpeaks(y_freq, 'SortStr', 'descend');

        % estima a frequencia fundamental
       onset_time = onset_time + subsampling_time;
       if length(peaklocs)>=8
         pitch_freq = mode(peaklocs(1:8));
       else
         pitch_freq = mode(peaklocs(1:end));
       end
       
       % energia do sinal
       energy = sum(subsample.^2);
       
       % verifica se a nota é igual ao do subsample anterior
       if abs(pitch_freq - previous_pitch_freq)/pitch_freq <= 0.03
                     
           % atualiza nota anterior
           adjusted_pitch = (pitch_freq + previous_pitch_freq)/2;
           [~,idx] = min(abs(frequence_reference_table(:,2)-adjusted_pitch));
           pitch = frequence_reference_table(idx,1);
           velocity = 50+energy; 
           duration = subsampling_time;
           onset = onset_time;
               
           % atualiza valor na matriz de dados MIDI
           nmat(end,4) = pitch;
           nmat(end,5) = velocity;  
           nmat(end,7) = nmat(end,7) + duration;           
       else
           % cria nova nota
           [~,idx] = min(abs(frequence_reference_table(:,2)-pitch_freq));
           pitch = frequence_reference_table(idx,1);
           velocity = 50+energy;
           duration = subsampling_time;
           onset = onset_time;
        
           % adiciona valor na matriz de dados MIDI
           nmat = [nmat; 0 0 3 pitch velocity onset duration];
       end    
       
       % atualiza valor para o próximo pacote
       subsample_begin = subsample_size*j + 1;
       previous_pitch_freq = pitch_freq;
    end
    
    % toca sinal MIDI
    %playsound(nmat)   
    
    % Salva sinal midi
    writemidi(nmat,[char(input_sound(i)) '.mid'])  
end

%melhorias - conseguir intensidade variavel durante a execução
%dividir as trilhas das notas de acordo com a mudança da frenquecia
