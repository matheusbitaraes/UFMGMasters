rm(list=ls())
library(mlbench)

trainadaline <- function(xin, yd, eta, tol, maxepocas, par){
  dimxin <- dim(xin)
  N<-dimxin[1]
  n<-dimxin[2]
  
  if (par==1){
    wt<-as.matrix(runif(n+1)-0.5)
    xin<-cbind(1, xin)
  } else {
    wt<-as.matrix(runif(n) -0.5)
  }
  nepocas <- 0 
  eepoca <- tol + 1
  
  evec <- matrix(nrow=1, ncol=maxepocas)
  while((nepocas < maxepocas) && (eepoca > tol)) {
    ei2 <- 0
    xseq <- sample(N)
    for(i in 1:N) {
      irand <- xseq[i]
      yhati<-1.0*((xin[irand,] %*% wt))
      ei <- yd[irand] - yhati
      dw <- eta*ei*xin[irand, ]
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

# carrega base de dados da boston housing
data("BostonHousing")

exec <- function(xall, yall, plot_title, with_plot=TRUE){
  
  maxx <- max(xall)
  xall <- xall/maxx
  
  maxy <- max(yall)
  yall <- yall/maxy
  
  xseq <- sample(506)
  xtrain <- as.matrix(xall[xseq[1:400],])
  ytrain <- as.matrix(yall[xseq[1:400],])
  xtest <- as.matrix(xall[xseq[401:506],])
  ytest <- as.matrix(yall[xseq[401:506],])
  
  retlist <- trainadaline(xtrain, ytrain, 0.1, 0.01, 1000, 1)
  w <- matrix(retlist[[1]], ncol=1)
  erro2 <- retlist[[2]]
  
  yhat <- (cbind(1, xtest) %*% w)
  yhattrain <- (cbind(1, xtrain) %*% w)
  
  if(with_plot){
    plot(1:length(ytest), ytest, type = "b", frame = FALSE, pch = 19, 
         col = "red", xlab = "x", ylab = "y")
    
    # Add a second line
    lines(matrix(yhat), pch = 18, col = "blue", type = "b", lty = 2)
    
    # Add a legend to the plot
    legend("topleft", legend=c("Teste", "Previsão"),
           col=c("red", "blue"), lty = 1:2, cex=0.8)
    
    title(plot_title)
  }
  
  return(erro2)
  
}

# olhar a sessão 3.4.6 das notas de aula
xall <- matrix(as.numeric(as.matrix(BostonHousing[,1:13])), nc=13)
yall <- matrix(as.numeric(as.matrix(BostonHousing[,14])), nc=1)

maxx <- max(xall)
xall <- xall/maxx

maxy <- max(yall)
yall <- yall/maxy

xyall <- cbind(xall,yall)
library("corrplot")
corrplot(cor(xyall), method="number", type="upper")


x_1 <- matrix(as.numeric(as.matrix(BostonHousing[,1:13])), nc=13)
x_2 <- x_1[,-9]
x_3 <- x_2[,-3]
err <- exec(x_1, yall, "Resultado para todas as variáveis")
err2 <- exec(x_2, yall, "Reusltado removendo variável 9")
err3 <- exec(x_3, yall, "Reusltado removendo variável 9 e 3")
mean(err)
mean(err2)
mean(err3)

multiple_execs <- function(x, y, num_execs){
  mean_error_list <- c()
  for (i in 1:num_execs) {
    print(sprintf("Running %s times", i))
    err <- exec(x, y, "", with_plot = FALSE)
    mean_error_list <- c(mean_error_list, mean(err))
  }
  return(mean_error_list)
}

mean_errs1 <- multiple_execs(x_1, yall, 50)
mean_errs2 <- multiple_execs(x_2, yall, 50)
mean_errs3 <- multiple_execs(x_3, yall, 50)


# Aplicação do kruskal e teste de nemenyi
errors <- data.frame("mean_quad_err_1" = mean_errs1, "mean_quad_err_2" = mean_errs2, "mean_quad_err_3" = mean_errs3)
errors_matrix <- as.matrix(errors)
tsutils::nemenyi(errors_matrix, conf.level=0.95,plottype="vmcb", sort=TRUE, main="Teste de Nemenyi (MSE)") # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"







