function [p, idxs] = paretofront(p)
% Filtra um conjunto de pontos P de acordo com a domin�ncia Pareto, ou
% seja, pontos que s�o dominados (ambos fracamente e fortemente) s�o
% filtrados.
%
% Entradas:
% - P:  Matriz N por D, onde N � o n�mero de pontos e D � o n�mero de
% elementos (objetivos) de cada ponto. 
%
% Sa�das:
% - P: Pareto-filtrado P
% - idxs: �ndices das solu��es n�o-dominadas
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
