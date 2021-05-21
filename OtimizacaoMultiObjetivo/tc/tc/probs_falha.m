function probs = probs_falha(equipamentos, fator_risco_planos, clusters, dt)
    t0 = equipamentos(:,2); % tempo em que os equipamentos estão operando desde
                         % sua data de instalação até o dia atual (anos)
    ni = clusters(equipamentos(:,3),2); % parametro de escala do modelo weibull
    beta = clusters(equipamentos(:,3),3); % parâmetro de forma do modelo weibull
    k = fator_risco_planos; % fator de risco associado aos planos de manutenção
    
    % modelo weibull
    F = @(t) 1 - exp(-(t./ni).^beta); 

    % probabilidade de falha
    Ft = 1 - exp(-((t0+k*dt)./ni).^beta);
    Ft0 = 1 - exp(-(t0./ni).^beta);
    
    probs = (Ft-Ft0)./(1-Ft0);
end