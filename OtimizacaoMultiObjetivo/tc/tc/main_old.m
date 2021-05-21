% leitura dos arquivos
clc
clear model

equipamentos = csvread('EquipDB.csv'); % [ID, t0, cluster, custo falha]
planos_manutencao = csvread('MPDB.csv'); % [ID, k - fator de risco, custo]
clusters = csvread('ClusterDB.csv'); % [ID, n, beta]
dt = 5; % anos

% vetor_de_solucoes = [2 3 2 1]; % este é um vetor de teste, considerando apenas os 4 primeiros equipamentos
% [custo_manutencao_total, custo_esperado_de_falha_total] = eval_custos(...
%     vetor_de_solucoes, equipamentos, planos_manutencao, clusters, dt);

% tamanho dos equipamentos
equipSize = size(equipamentos,1);


% x será binário, do tamanho [n_equipamentos x n_planos]
% a soma das colunas de x deve ser 0<=x<=1, ou seja, cada um com apenas um
% plano selecionado



% Initialize model
model.modelsense  = 'min';
model.modelname   = 'multiobj';

% Set variables and constraints
model.vtype       = repmat('i', 3*equipSize,1);
model.lb          = zeros(3*equipSize, 1);
model.ub          = ones(3*equipSize, 1);
model.A           = sparse([eye(equipSize) eye(equipSize) eye(equipSize)]);
model.rhs         = ones(equipSize,1);
model.sense       = '=';
% model.constrnames = {'Budget'};

% custo de manutenção total
custos_por_plano = repmat(planos_manutencao(:,3)',equipSize,1);
custos_reshaped = reshape(custos_por_plano,[3*equipSize,1]);
custos_reshaped_scaled = custos_reshaped/max(custos_reshaped);
model.multiobj(1).objn     = custos_reshaped;
model.multiobj(1).weight    = 1.0;
model.multiobj(1).name     = sprintf('Custo de manutenção total');

% custo esperado de falha
fator_risco_planos = repmat(planos_manutencao(:,2)',equipSize,1);
pf = probs_falha(equipamentos, fator_risco_planos, clusters, dt);
custo_esp_falha = pf.*equipamentos(:,4);
custo_esp_falha_reshaped = reshape(custo_esp_falha,[3*equipSize,1]);
custo_esp_falha_reshaped_scaled = custos_reshaped/max(custos_reshaped);
model.multiobj(2).objn      = custo_esp_falha_reshaped;
model.multiobj(2).weight    = 1.0;
model.multiobj(2).name      = sprintf('Custo esperado de falha');


% Save model
% gurobi_write(model,'multiobj_m.lp')

% Set parameters
params.PoolSolutions = 100;

% Optimize
result = gurobi(model);

% Capture solution information
if ~strcmp(result.status, 'OPTIMAL')
    fprintf('Optimization finished with status %d, quit now\n', result.status);
    return;
end

% Print best solution
% fprintf('Selected elements in best solution:\n');
% for j = 1:equipSize
%     if result.x(j) >= 0.9
%         fprintf('%s ', model.varnames{j});
%     end
% end
% fprintf('\n');

% Print all solution objectives and best furth solution
if isfield(result, 'pool') && ~isempty(result.pool)
    solcount = length(result.pool);
    fprintf('Number of solutions found: %d\n', solcount);
    fprintf('Objective values for first %d solutions:\n', solcount);
    for m = 1:2
        fprintf('  %s:', model.multiobj(m).name);
        for k = 1:solcount
            fprintf('  %3g', result.pool(k).objval(m));
        end
        fprintf('\n');
    end
    fprintf('\n');
else
    fprintf('Number of solutions found: 1\n');
    fprintf('Solution 1 has objective values:');
    for k = 1:2
        fprintf('  %g', result.objval(k));
    end
    fprintf('\n');
end
% 
% % resultado: curva pareto ficticia
% custos_manutencao = (1:60);
% custos_falha = (1:60);
% 
% figure(1)
% plot(custos_manutencao, custos_falha, '*')
