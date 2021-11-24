#perceptron
yperceptron <- function(xvec, w){
  # xvec: vetor de entrada
  # w: vetor de pesos
  # yp: resposta do Perceptron
  u<-xvec %*% w
  y<-1.0*(u>=0)
  
  return(as.matrix(y))
}