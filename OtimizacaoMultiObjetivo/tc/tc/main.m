% leitura dos arquivos
clc
clear model

equipamentos = csvread('EquipDB.csv'); % [ID, t0, cluster, custo falha]
planos_manutencao = csvread('MPDB.csv'); % [ID, k - fator de risco, custo]
clusters = csvread('ClusterDB.csv'); % [ID, n, beta]
dt = 5; % anos

% tamanho dos equipamentos
equipSize = size(equipamentos,1);

% custo de manutenção total
custos_por_plano = repmat(planos_manutencao(:,3)',equipSize,1);
custos_reshaped = reshape(custos_por_plano,[3*equipSize,1]);

% custo esperado de falha
fator_risco_planos = repmat(planos_manutencao(:,2)',equipSize,1);
pf = probs_falha(equipamentos, fator_risco_planos, clusters, dt);
custo_esp_falha = pf.*equipamentos(:,4);
custo_esp_falha_reshaped = reshape(custo_esp_falha,[3*equipSize,1]);

%probabilidade esperada de falha (criterio extra para tomada de decisão)
pf_reshaped = reshape(pf,[3*equipSize,1]);

% Initialize model
model.modelsense  = 'min';
model.modelname   = 'multiobj';

% Set variables and constraints
model.vtype       = repmat('b', 3*equipSize,1);
model.lb          = zeros(3*equipSize, 1);
model.ub          = ones(3*equipSize, 1);
model.A           = sparse([custos_reshaped' ; [eye(equipSize) eye(equipSize) eye(equipSize)]]);
model.sense       = [ '<' repmat('=', 1,equipSize)];
model.obj     = custo_esp_falha_reshaped;
model.name     = sprintf('Custo de manutenção total');

step = 1; % variacao do epsilon

maior_custo = sum(custos_por_plano(:,3));
menor_custo = sum(custos_por_plano(:,1));


figure()
title('Fronteira pareto')
xlabel('Custo de manutenção total')
ylabel('Custo esperado de falha')
hold on

i = 1;
pareto_front = [];
optimal_x = [];
decision_variables = [];
for epsilon = menor_custo:step:maior_custo 
    model.rhs = [epsilon; ones(equipSize,1)]; % atualiza epsilon
    % Otimiza
    result = gurobi(model);

    % verifica solução
    if strcmp(result.status, 'OPTIMAL')
        custo_manutencao = sum(result.x .* custos_reshaped);
        prob_esperada_falha = sum(result.x .* pf_reshaped);
        plot(custo_manutencao, result.objval, '*b')
        pareto_front(i,1) = custo_manutencao;
        pareto_front(i,2) = result.objval;
        decision_variables(i,3) = prob_esperada_falha;
        optimal_x(i,:) = sum([1 2 3].*reshape(result.x,equipSize,3),2)';
        i = i + 1;
    end
end
decision_variables(:,1:2) = pareto_front;
unique_values = unique([pareto_front optimal_x], 'rows');
pareto_front = unique_values(:,1:2);
optimal_x = unique_values(:,3:502);

save ('optimization_data_eps_01','pareto_front','optimal_x', 'decision_variables')

% exporta para csv
csvwrite('BitaraesDuartePereira.csv', optimal_x)
