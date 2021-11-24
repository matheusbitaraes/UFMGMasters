#perceptron
yperceptron <- function(xvec, w, pol){
  # xvec: vetor de entrada
  # w: vetor de pesos
  # yp: resposta do Perceptron
  
  u<- xvec %*% w[2:length(w)] + pol * w[1]
  y<-1.0*(u>=0)
  
  return(as.matrix(y))
}