function v = hypervolume(P,r,N)

%HYPERVOLUME    Indicador de hipervolume � dado como uma medida da
%fronteira Pareto estimada. V= hypervolume(P,r,N) retorna uma estimativa do
%hipervolume dominado (em porcentagem) pela fronteira Pareto aproximada
%estabelecida P (n por d) e � delimitado pelo ponto de refer�ncia
%n�o-ut�pico R (1 por d). A estima��o at� N (definindo N = 1000 pontos
%aleat�rios uniformemente distribu�dos)
% V = HYPERVOLUMN(P,R,C) usa os pontos de teste especificados em C (N por
% d).

% um exemplo rand�mico

%F = (randn(100,3) + 5).^2; 

% limite superior do conjunto de dados

%r = max(F); 

% Aproxima��o do conjunto Pareto

% P = paretofront(F);
% hypervolume
% v = hypervolume(P,r,100000);

% Verifica��o de entrada e sa�da

% Check input and output
error(nargchk(2,3,nargin));
error(nargoutchk(0,1,nargout));
P=P*diag(1./r);
[n,d]=size(P);
if nargin<3
    N=1000;
end
if ~isscalar(N)
    C=N;
    N=size(C,1);
else
    C=rand(N,d);
end
lB=[0,0];
fDominated=false(N,1);
fcheck=all(bsxfun(@gt, C, lB),2);
figure
for k=1:n
    if any(fcheck)
        f=all(bsxfun(@gt, C(fcheck,:), P(k,:)),2);
        fDominated(fcheck)=f;
        fcheck(fcheck)=~f;
    end
end
scatter(P(:,1), P(:,2))
hold on

scatter(C(:,1),C(:,2))
scatter(C(fcheck,1),C(fcheck,2))
v=sum(fDominated)/N;

