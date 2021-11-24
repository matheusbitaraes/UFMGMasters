# aplicação do perceptron para separação
trainperceptron <- function(xin, yd, eta, tol, maxepocas, par){
  # xin: matrin N x n com dados de entrada
  # yd: rótulos de saída
  # eta: passo do treinamento
  # tol: tolerancia de erro
  # maxepocas: numero maximo de iterações
  # par: parametro de entrada
  # par=0 ==> xin tem dimensão n+1 e já inclui entrada correspondente ao temo de polarização
  # par =1 ==> xin tem dimensão n e não inclui entrada corresponde ao termo de polarização, que deve ser adicionado dentro da função
  
  dimxin<-dim(xin)
  N<-dimxin[1]
  n<-dimxin[2]
  
  if(par==1){
    wt<-as.matrix(runif(n+1) - 0.5)
    xin<-cbind(-1,xin)
  }
  else{
    wt<-as.matrix(runif(n)-0.5)
  }
  nepocas<-0
  eepoca<-tol+1
  evec<-matrix(nrow=1,ncol=maxepocas)
  
  while((nepocas<maxepocas && eepoca>tol)){
    ei2<-0
    xseq<-sample(N)
    for(i in 1:N){
      irand<-xseq[i]
      yhati<-1.0*((xin[irand,] %*% wt)>=0)
      ei<-yd[irand]-yhati
      dw<-eta*ei*xin[irand,]
      wt<-wt+dw
      ei2<-ei2+ei*ei
    }
    nepocas<-nepocas+1
    evec[nepocas]<-ei2/N
    eepoca<-evec[nepocas]
  }
  retlist<-list(wt, evec[1:nepocas])
  return(retlist)
}
