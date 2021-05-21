function [p, idxs] = paretofront(p)
% Filtra um conjunto de pontos P de acordo com a dominância Pareto, ou
% seja, pontos que são dominados (ambos fracamente e fortemente) são
% filtrados.
%
% Entradas:
% - P:  Matriz N por D, onde N é o número de pontos e D é o número de
% elementos (objetivos) de cada ponto. 
%
% Saídas:
% - P: Pareto-filtrado P
% - idxs: índices das soluções não-dominadas
%
% Examplo:
% p = [1 1 1; 2 0 1; 2 -1 1; 1, 1 0];
% [f, idxs] = paretofront(p)
% f = [1 1 1, 2 0 1] 
% idxs = [1;2]

[i, dim] = size(p);
idxs = [1: i]'; 
while i >= 1
    old_size = size(p,1);
    indices = sum(bsxfun(@ge, p(i,:),p),2) == dim; 
    indices(i) = false;
    p(indices,:) = []; 
    idxs(indices) = []; 
    i = i - 1 - (old_size - size(p,1)) + sum(indices(i:end));
end

end
