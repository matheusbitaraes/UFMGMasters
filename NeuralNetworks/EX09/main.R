rm(list=ls())
library("RSNNS")

# sech2 <-function(u){
#   return(((2/(exp(u)+exp(-u)))*(2/(exp(u)+exp(-u)))))
# }

N<-45
x<-runif(N,0,2*pi)
y<-sin(x) + rnorm(N,0,0.1)

plot(x,y,type="p", xlab="x", ylab="y", col="blue")

xtest <- seq(0, 2*pi,.001)
ytest <-sin(xtest)
lines(xtest,ytest,type="l", xlab="x", ylab="y", col="red")


model <- mlp(x, y, size=3, maxit=2000, initFunc = "Randomize_Weigths",
             initFuncParams = c(-0.3,0.3), learnFunc = "Rprop",
             learnFuncParams = c(0.1, 0.1), updateFunc = "Topological_Order",
             updateFuncParams = c(0), hiddenActFunc = "Act_Logistic",
             shufflePatterns = TRUE, linOut = TRUE)

yhat<-predict(model, as.matrix(xtest))
lines(xtest,yhat,type="l", xlab="x", ylab="y", col="green")
legend(x="topright",legend=c("Train", "Test", "Predicted"), col=c("blue", "red", "green"), lty = c(1, 1))
title("Resposta do modelo MLP de 3 neuronios")

error <- model$IterativeFitError[model$maxit]
print(error)
plot(c(1:model$maxit),model$IterativeFitError,type="l", xlab="iterations", ylab="error", col="blue")
title("Erros por iteração")


es<-c(0.8210759, 2.474404, 1.87387, 0.4599661, 0.4694107)
print(mean(es))
print(sd(es))
# x<-matrix(c(0,0,0,1,1,0,1,1),ncol=2, byrow=T)
xatual<-matrix(nrow=2, ncol=1)
# y<-matrix(c(-1,1,1,-1,1,-1,-1,1),ncol=2,byrow=T)
Z<-matrix(runif(6)-.5, ncol=1,nrow=3)
W<-matrix(runif(6)-.5, ncol=1,nrow=3)

tol<-0.1
eta<-0.01
maxepocas <- 2000
nepocas<-0
eepoca<-tol+1
N<-length(x)
evec<-matrix(nrow=maxepocas,ncol=1)

while((nepocas<maxepocas) && (eepoca > tol)){
  ei2 <- 0
  xseq <- sample(N)
  for (i in 1:N){
    irand<-xseq[i]
    
    xatual[1, 1]<-x[irand]
    xatual[2, 1]<-1
    
    yatual<-y[irand]
    U<-t(xatual)%*%Z
    H<-tanh(U)
    Haug<-cbind(H,1)
    O<-Haug%*%W
    yhat<-tanh(O)
    
    e<-yatual-yhat
    flinhao<-sech2(O)
    dO<-e*flinhao
    
    Wminus<-W[-2,]
    ehidden<-dO %*% t(Wminus)
    flinhaU<-sech2(U)
    dU<-ehidden * flinhaU
    
    W<-W+eta*(t(Haug)%*%dO)
    Z<-Z+eta*(xatual%*%dU)
    ei2<-ei2+(e%*%t(e))
  }
  
  nepocas<-nepocas+1
  evec[nepocas]<-ei2/N
  eepoca<-evec[nepocas]
}
plot(evec[1:nepocas],type='l')


xatual[(1:2),1]<-x[2,(1:2)]
xatual[3,1]<-1
yatual<-y[2,(1:2)]
U<-t(xatual)%*%Z
H<-tanh(U)
Haug<-cbind(H,1)
O<-Haug%*%W
yhat<-tanh(O)
e<-yatual-yhat
flinhaO<-sech2(O)
dO<-e
