function prob = prob_falha(equipamento, plano_manutencao, clusters, dt)
    t0 = equipamento(2); % tempo em que o equipamento está operando desde
                         % sua data de instalação até o dia atual (anos)
    ni = clusters(equipamento(3),2); % parametro de escala do modelo weibull
    beta = clusters(equipamento(3),3); % parâmetro de forma do modelo weibull
    k = plano_manutencao(2); % fator de risco associado ao plano de manutenção
    
    % modelo weibull
    F = @(t) 1 - exp(-(t/ni)^beta); 

    % probabilidade de falha
    prob = (F(t0+k*dt)-F(t0))/(1-F(t0));
end