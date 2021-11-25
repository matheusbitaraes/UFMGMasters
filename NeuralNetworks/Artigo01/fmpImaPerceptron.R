#perceptron
y_voted_percetron <- function(xvec, w, c, pol){
  # xvec: vetor de entrada
  # w: vetor de pesos
  # yp: resposta do Perceptron
  
  s <- 0
  for (i in 1:nrow(w)){
    u <- xvec %*% w[i, 2:ncol(w)] + pol * w[i, 1]
    s <- s + c[i] * sign(u)
  }
  return(sign(s))
}

# aplicação do perceptron para separação
train_voted_perceptron <- function(xin, yd, eta, tol, maxepocas, par){
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
  
  
  wt <- matrix(0, ncol=ncol(xin), nrow=N)
  c <- matrix(0, ncol=1, nrow=N)
  while((nepocas<maxepocas && eepoca>tol)){
    ei2<-0
    xseq<-sample(N)
    k <- 1
    for(i in 1:N){
      irand<-xseq[i]
      yhati<-1.0*((xin[irand,] %*% wt[k,])>=0)
      ei<-yd[irand]-yhati
      if (ei == 0){ # acertou
        c[k] <- c[k] + 1
      } else { # errou
        dw<-eta*ei*xin[irand,]
        wt[k + 1,] <- wt[k,] + dw
        c[k + 1,] <- 1
        ei2<-ei2+ei*ei
        k <- k + 1
      }
    }
    nepocas<-nepocas+1
    evec[nepocas]<-ei2/N
    eepoca<-evec[nepocas]
  }
  retlist<-list(wt[1:k,], c[1:k,], evec[1:nepocas])
  return(retlist)
}

eval_voted_perceptron <- function(X, Y, should_plot_matrix=FALSE){
  # separação em treinamento e teste
  nc <- nrow(X)
  suffled_indexes <- sample(nc)
  train_size <- floor(nc * 0.70)
  x_train <- X[suffled_indexes[1:train_size],]
  y_train <- Y[suffled_indexes[1:train_size]]
  x_test <- X[suffled_indexes[(train_size+1):nc],]
  y_test <- Y[suffled_indexes[(train_size+1):nc]]
  
  # treinamento do voted perceptron
  sol <- train_voted_perceptron(x_train, y_train, 0.01, 0.01, 50, 1)
  v <- sol[[1]]
  c <- sol[[2]]
  err <- c(0)
  
  # acurácia
  ypred <- matrix(0,nrow=dim(x_test)[1], ncol=1)
  
  for (i in 1:dim(x_test)[1]){
    s <- y_voted_percetron(x_test[i,], v, c, -1)
    ypred[i] <- s
  }
  # matriz de confusão
  lvs <- c("-1", "1")
  truth <-  factor(y_test, levels = rev(lvs))
  pred <- factor(ypred, levels = rev(lvs))
  
  xtab <- table(pred, truth)
  cm <- confusionMatrix(xtab)
  if (should_plot_matrix){
    print(cm)
  }
  return (c(cm$overall[1], err[length(err)]))
}


### FUNÇÕES PARA TESTAR O MÉTODO ###

# criação de dataset com duas distribuições normais
xc1 <- matrix(rnorm(nc * 2), ncol = 2)*s1 + t(matrix((c(2,2)),ncol=nc,nrow=2))
xc2 <- matrix(rnorm(nc * 2), ncol = 2)*s2 + t(matrix((c(4,4)),ncol=nc,nrow=2))
xc1 <- cbind(xc1, rep(0, times = nc/2))
xc2 <- cbind(xc2, rep(1, times = nc/2))
X <- rbind(xc1, xc2)

# separação em treinamento e teste
suffled_indexes <- sample(nt)
train_size <- nt * 0.7
X_train <- X[suffled_indexes[1:train_size], cbind(1,2)]
y_train <- X[suffled_indexes[1:train_size], 3]
X_test <- X[suffled_indexes[(train_size+1):nt], cbind(1,2)]
y_test <- X[suffled_indexes[(train_size+1):nt], 3]


# treinamento do perceptron
sol <- train_voted_perceptron(X_train, y_train, 0.01, 0.01, 50, 1)
w <- sol[[1]]
c <- sol[[2]]

# acurácia
ypred <- matrix(0,nrow=dim(X_test)[1], ncol=1)

for (i in 1:dim(X_test)[1]){
  s <- y_voted_percetron(X_test[i,], w, c,-1)
  ypred[i] <- s
}
correct <- length(ypred[ypred==y_test])
total <- length(ypred)

acc <- correct/total

print(c("acc:", acc))

# matriz de confusão
lvs <- c("1", "0")
truth <-  factor(y_test, levels = rev(lvs))
pred <- factor(ypred, levels = rev(lvs))

xtab <- table(pred, truth)
# load Caret package for computing Confusion matrix
library(caret) 
confusionMatrix(xtab)

seqi<-seq(0,6, 0.1)
seqj<-seq(0,6, 0.1)
M<-matrix(0,nrow=length(seqi), ncol=length(seqj))
ci<-0
for(i in seqi){
  ci<-ci+1
  cj<-0
  for(j in seqj){
    cj<-cj+1
    x<-c(i,j)
    M[ci,cj]<-y_voted_percetron(x, w, c, -1)
  }
}

plot(xc1[,1], xc1[,2], col='red', xlim = c(0,6), ylim=c(0,6), xlab='x_1', ylab='x_2')
par(new=T)
plot(xc2[,1], xc2[,2], col='blue', xlim = c(0,6), ylim=c(0,6), xlab='', ylab='')
par(new=T)
contour(seqi, seqj, M, xlim=c(0,6), ylim=c(0,6))
title("Região de Separação (2D)")

persp3D(seqi,seqj,M,counter=T, theta=55, phi=30, r=40, d=0.1, expand=0.5,
        ltheta=90, lphi=180, shade=0.4, ticktype="detailed", nticks=5)
title("Região de Separação (3D)")