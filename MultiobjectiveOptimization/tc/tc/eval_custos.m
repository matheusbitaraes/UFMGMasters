function [custo_manutencao_total, custo_esp_falha_total] = eval_custos(vetor_de_solucoes, equipamentos, planos_manutencao, clusters, dt)
%  - vetor_de_solucoes: vetor com os planos de manutencao para cada
% equipamento. ex. [3 1 1 3 2 3 2 1 ...] significa que, para o equipamento
% 1, foi aplicado o plano de manutencao 1, e assim sucessivamente
%
%  - equipamentos: dados do EquipDB.csv
% 
%  - planos_manutencao: dados do MPBD.csv
% 
%  - clusters: dados do ClusterDB.csv

    custo_manutencao_total = sum(planos_manutencao(vetor_de_solucoes,3));
    custo_esp_falha_total = 0; % inicialização do custo de falha
    
    % iterando em cada posicao do vetor de solucoes para calcular
    % propabilidade de falha * custo de falha do equipamento 
    for i = 1:length(vetor_de_solucoes) 
        plano_manutencao = planos_manutencao(vetor_de_solucoes(i),:);
        p = prob_falha(equipamentos(i,:), plano_manutencao, clusters, dt);
        custo_esp_falha_total = custo_esp_falha_total + p*equipamentos(i, 4);
    end
end