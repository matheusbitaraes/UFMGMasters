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
    k <- 1
    xseq<-sample(N)
    for(i in 1:N){
      irand<-xseq[i]
      yhati<-1.0*((xin[irand,] %*% wt[k,])>=0)
        if( yd[irand] == yhati){
          c[k] <- c[k] + 1
        } else {
          ei<-yd[irand]-yhati
          dw<-eta*ei*xin[irand,]
          wt[k+1]<-wt[k]+dw
          ei2<-ei2+ei*ei
          c[k + 1,] <- 1
          k <- k + 1
        }
    }
    nepocas<-nepocas+1
    evec[nepocas]<-ei2/N
    eepoca<-evec[nepocas]
    k <- k + 1
  }
  retlist<-list(wt[1:k,], c[1:k,], evec[1:nepocas])
  return(retlist)
  
  #   ei2<-0
  #   k <- 1
  #   c <- matrix(0, ncol=1, nrow=N)
  #   xseq<-sample(N)
  #   for(i in 1:N){
  #     irand<-xseq[i]
  #     pred_y <-1.0*((xin[irand,] %*% wt[k,])>=0)
  #     expected_y <- yd[irand]
  #     ei<-expected_y-pred_y
  #     ei2<-ei2+ei*ei
  #     if( expected_y == pred_y){
  #       c[k] <- c[k] + 1
  #     } else {
  #       dw<-expected_y*xin[irand,]
  #       wt[k + 1,] <- wt[k,] + dw
  #       c[k + 1] <- 1
  #       k <- k + 1
  #     }
  #   }
  #   nepocas<-nepocas+1
  #   evec[nepocas]<-ei2/N
  #   eepoca<-evec[nepocas]
  # }
  # 
  # retlist<-list(wt[k-1:k,], c[k-1:k], eepoca)
  # return(retlist)
}

eval_voted_perceptron <- function(X, Y, should_plot_matrix=FALSE){
  # Y deve ser -1 e 1:
  Y[Y==0] <- -1
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


# script para testar os métodos
library(caret) 

multiple_eval <- function(X, Y, num_eval){
  
  perc_acc <- matrix(0, nrow=num_eval, ncol=1)
  perc_time <- matrix(0, nrow=num_eval, ncol=1)
  fmp_acc <- matrix(0, nrow=num_eval, ncol=1)
  fmp_time <- matrix(0, nrow=num_eval, ncol=1)
  vp_acc <- matrix(0, nrow=num_eval, ncol=1)
  vp_time <- matrix(0, nrow=num_eval, ncol=1)
  for (i in 1:num_eval){
    print(sprintf("Time for voted perceptron - %s", i))
    vp_start_time <- Sys.time()
    s3 <- eval_voted_perceptron(X, Y, should_plot_matrix=FALSE)
    vp_end_time <- Sys.time()
    
    vp_acc[i] <- s3[1]
    vp_time[i] <- vp_end_time - vp_start_time
    
    print(sprintf("voted perc: %s \n\n", s3[1]))
  }
  
  return(list(data.frame("voted_perceptron" = vp_acc),
              data.frame("voted_perceptron" = vp_time)))
}

eval <- function(X, Y, name, num_exec){
  results <- multiple_eval(X, Y, num_exec)
  accs <- results[[1]]
  times <- results[[2]]
  
  par(mfrow=c(2,2))
  
  # Boxplot
  par(mar=c(2,10,1,1))
  boxplot(accs, data=data,
          las = 2,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = sprintf("Acurácias (%s)",name))
  
  # Boxplot
  par(mar=c(2,10,1,1))
  boxplot(times, data=data,
          las = 2,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = sprintf("Tempo (%s)",name))
  
  # Aplicação do kruskal e teste de nemenyi
  accs_matrix <- as.matrix(accs)
  tsutils::nemenyi(accs_matrix, conf.level=0.95,plottype="vmcb", sort=TRUE, main="Teste de Nemenyi (Acurácias)") # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"
  
  # Aplicação do kruskal e teste de nemenyi
  time_matrix <- as.matrix(times)
  tsutils::nemenyi(time_matrix, conf.level=0.95,plottype="vmcb", sort=TRUE, main="Teste de Nemenyi (Tempo)") # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"
}

nc = 500
xc1 <- matrix(0.35 * rnorm(nc) + 2, ncol = 2)
xc2 <- matrix(0.35 * rnorm(nc) + 4, ncol = 2)
xc1 <- cbind(xc1, rep(0, times = nc/2))
xc2 <- cbind(xc2, rep(1, times = nc/2))
X <- rbind(xc1, xc2)
plot(xc1[,1], xc1[,2], col="blue", xlim=c(1, 5), ylim=c(1, 5)) # plot do dataset
points(xc2[,1], xc2[,2], col="red", pch=1) # plot do dataset
title(main="Two Gaussians Dataset")
name <- "Two Gaussian"

eval(X[,1:2], X[,3], name, 50)