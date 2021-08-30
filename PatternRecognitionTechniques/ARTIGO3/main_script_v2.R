rm(list=ls())
graphics.off()

library(mlbench)
library(e1071)
library(caret)
library(kernlab)
library(GGClassification)
library(tsutils)
library(randomForest)

# MAIN
multiple_evaluations <- function(X, Y, num_clusters, num_eval, svm_kernel='rbfdot' ){
  nclusters <- num_clusters
  nm_acc <- c()
  svm_acc <- c()
  gg_acc <- c()
  nm2_acc <- c()
  nm_time <- c()
  svm_time <- c()
  gg_time <- c()
  nm2_time <- c()
  
  # FUNÇÕES DE SUPORTE
  get_cluster <- function(x, km){
    centers <- km$centers
    dists <- rowSums(x - centers)^2 
    cluster_id <- which(dists==min(dists))
    return (cluster_id)
  }
  
  pred_func <- function (x, cluster_model, km, nclusters, svm_option, gg_option){
    num_col <- ncol(x)
    cluster_col <- num_col + 1
    result_col <- num_col + 2
    x <- cbind(x, 0)
    
    # atribui clusters
    for (i in 1:nrow(x)){
      cluster_id <- get_cluster(x[i,], km)
      x[i, cluster_col] <- cluster_id
    }
    x <- cbind(x, 0)
    # print(nclusters)
    for (cid in 1:nclusters){
      if (is.null(nrow(x[x[, cluster_col]==cid,]))){
        result_count <- 1
      } else{
        result_count <- nrow(x[x[, cluster_col]==cid,])
      }
      # print(sprintf("%s -> %s",cid, result_count))
      if(result_count > 0){
        model_name <- cluster_model[cid]
        # print(sprintf("Cluster: %s - model: %s",cid, model_name))
        if (model_name == "GG"){
          input <- matrix(x[x[, cluster_col]==cid, 1:num_col], result_count, num_col)
          x[x[, cluster_col]==cid, result_col] <- GGClassification::predict(gg_option, input)
        }else if (model_name == "SVM"){
          input <- matrix(x[x[, cluster_col]==cid, 1:num_col], result_count, num_col)
          x[x[, cluster_col]==cid, result_col] <- kernlab::predict(svm_option, input, type='response')
        }
      }
      # print("-")
    }
    return(x[, result_col])
  }
  
  eval_model <- function(x, y, model, n_exec, type){
    n <- nrow(x)
    eval_size <- floor(n * 0.99)
    
    acc <- c()
    for (i in 1:n_exec){
      suffled_indexes <- sample(n)
      x_ <- x[suffled_indexes[1:eval_size],]
      y_ <- y[suffled_indexes[1:eval_size]]
      if(type=="GG"){
        pred <- GGClassification::predict(model, x_)
      }else if(type=="SVM"){
        pred <- kernlab::predict(model, x_)
      }
      acc <- c(acc, sum(pred == y_)/length(y_))
    }
    return (mean(acc))
  }
  
  for(neval in 1:num_eval){
    nc <- nrow(X)
    suffled_indexes <- sample(nc)
    train_size <- floor(nc * 0.80)
    x_train <- X[suffled_indexes[1:train_size],]
    y_train <- Y[suffled_indexes[1:train_size]]
    x_test <- X[suffled_indexes[(train_size+1):nc],]
    y_test <- Y[suffled_indexes[(train_size+1):nc]]
    
    # clusterizar o dataset
    nm_start_time <- Sys.time()
    km <- kmeans(as.matrix(x_train), nclusters, iter.max=50)
    clusters <- as.matrix(km[[1]])
    
    # treinamento de svm em clusteres
    svm_option <- kernlab::ksvm(x_train, y_train, type='C-svc', kernel=svm_kernel)
    gg_option <- GGClassification::model(x_train, y_train)
    
    cluster_model <- c()
    for (i in 1:nclusters){
      # print(sprintf("treinando cluster %s", i))
      x_ <-x_train[clusters == i,]
      y_ <-y_train[clusters == i]
      acc_svm <- eval_model(x_, y_, svm_option, n_exec=1, type="SVM")
      print(sprintf("SVM (acc): %s", acc_svm))
      acc_gg <- eval_model(x_, y_, gg_option, n_exec=1, type="GG")
      print(sprintf("GG (acc): %s", acc_gg))
      print("--")
      if (acc_gg > acc_svm){
        cluster_model <- c(cluster_model, "GG")
      }else{
        cluster_model <- c(cluster_model, "SVM")
      }
    }
    
    y_pred <- pred_func(x=x_test, cluster_model, km=km, nclusters=nclusters, svm_option, gg_option)
    nm_end_time <- Sys.time()
    # cm_nm <- confusionMatrix(as.factor(y_pred), as.factor(y_test))
    
    # SVM
    svm_start_time <- Sys.time()
    svmtrein <- kernlab::ksvm(x_train, y_train, type='C-svc', kernel=svm_kernel)
    predSVM <- kernlab::predict(svmtrein, x_test, type='response')
    svm_end_time <- Sys.time()
    # cm_svm <- confusionMatrix(as.factor(predSVM), as.factor(y_test))
    
    #GG
    gg_start_time <- Sys.time()
    mdl <- GGClassification::model(x_train, y_train)
    y_pred_gg <- predict(mdl, x_test)
    gg_end_time <- Sys.time()
    # cm_gg <- confusionMatrix(as.factor(y_pred_gg), as.factor(y_test))
    
    # rf
    nm2_start_time <- Sys.time()
    rf <- randomForest(x_train, y_train)
    y_pred_nm2 <- round(kernlab::predict(rf, x_test))
    nm2_end_time <- Sys.time()
    
    nm_acc <- c(nm_acc, sum(y_pred == y_test)/length(y_test))
    svm_acc <- c(svm_acc, sum(predSVM == y_test)/length(y_test))
    gg_acc <- c(gg_acc, sum(y_pred_gg == y_test)/length(y_test))
    nm2_acc <- c(nm2_acc, sum(y_pred_nm2 == y_test)/length(y_test))
    nm_time <- c(nm_time, nm_end_time - nm_start_time)
    svm_time <- c(svm_time, svm_end_time - svm_start_time)
    gg_time <- c(gg_time, gg_end_time - gg_start_time)
    nm2_time <- c(nm2_time, nm2_end_time - nm2_start_time)
    
    print("EVALUATION")
    print(sprintf("cluster model: %s (%s) ",sum(y_pred == y_test)/length(y_test),
                  nm_end_time - nm_start_time))
    print(sprintf("SVM: %s ", sum(predSVM == y_test)/length(y_test),
                  svm_end_time - svm_start_time))
    print(sprintf("ggClassification: %s ",sum(y_pred_gg == y_test)/length(y_test),
                  gg_end_time - gg_start_time))
    print(sprintf("committee model: %s ",sum(y_pred_nm2 == y_test)/length(y_test),
                  nm2_end_time - nm2_start_time))
    print("----------------------------------------------------")
  }
  return(list(data.frame("nm" = nm_acc, "svm" = svm_acc, "gg" = gg_acc, "nm2" = nm2_acc),
              data.frame("nm" = nm_time, "svm" = svm_time, "gg" = gg_time, "nm2" = nm2_time)))
}

evaluate_dataset <- function(X, Y, name, nclusters, num_exec){
  results <- multiple_evaluations(X, Y, nclusters, num_exec)
  accs <- results[[1]]
  times <- results[[2]]
  
  par(mfrow=c(2,2))
  
  # Boxplot
  par(mar=c(4,10,1,1))
  boxplot(accs, data=data,
          las = 2,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = sprintf("Acurácias (%s)",name))
  
  # Boxplot
  par(mar=c(4,10,1,1))
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

######################################### TWO GAUSSIAN DATASET ######################################### 
nclusters <- 5
nc = 500
xc1 <- matrix(0.45 * rnorm(nc) + 2.5, ncol = 2)
xc2 <- matrix(0.45 * rnorm(nc) + 3.5, ncol = 2)
xc1 <- cbind(xc1, rep(0, times = nc/2))
xc2 <- cbind(xc2, rep(1, times = nc/2))
X <- rbind(xc1, xc2)
plot(xc1[,1], xc1[,2], col="blue", xlim=c(1, 5), ylim=c(1, 5)) # plot do dataset
points(xc2[,1], xc2[,2], col="red", pch=1) # plot do dataset
title(main="Two Gaussians Dataset")
name <- "Two Gaussian"

km <- kmeans(as.matrix(X[,1:2]), nclusters, iter.max=50)
clusters <- as.matrix(km[[1]])
# plot cluster centers
points(km$centers, col="green", pch=16)

# plot samples
pchs <- c()
names <- c()
for (i in 1:nclusters){
  pch = 5 + i
  points(X[clusters == i,1], X[clusters == i,2], pch=pch)
  pchs <- c(pchs, pch)
  names <- c(names, sprintf("Cluster %s", i))
}
legend("topleft", legend=names, pch=pchs)

evaluate_dataset(X[,1:2], X[,3], name, nclusters, 100)

######################################### IRIS DATASET ######################################### 
name <- "Iris Dataset"
iris.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris <- read.csv(iris.url, header=FALSE)
# pre processamento
iris$V5[iris$V5 == 'Iris-setosa'] <- 'Iris-versicolor' #juntando classe 0 com classe 1
iris$V5 <- factor(iris$V5)
iris$V5 <- as.factor(iris$V5)
names(iris)[5] <- 'y'
# plot(iris)

# converter em matrix
mat <- data.matrix(iris)
X <- scale(mat[, 1:4], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 5]

evaluate_dataset(X, Y, name, 5, 100)

######################################### WINE DATASET ######################################### 
name <- "Wine Dataset"
wine.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine <- read.csv(wine.url, header=FALSE)
# pre-processamento
wine <- wine[1:130,]# filtra apenas dois tipos de vinho, para que fique binario
wine$V1[wine$V1 == 2] <- -1
wine$V1 <- as.factor(wine$V1)
names(wine)[1] <- 'y'

# converter em matrix
mat <- data.matrix(wine)
X <- scale(mat[, 2:14], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 1]

evaluate_dataset(X, Y, name, 5, 100)

######################################### PimaIndiansDiabetes ######################################### 
# load the dataset
name <- "Pima Indians"
pima <- data(PimaIndiansDiabetes)
names(PimaIndiansDiabetes)[9] <- "y"

# converter em matrix
mat <- data.matrix(PimaIndiansDiabetes)
X <- scale(mat[, 1:8], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 9]

evaluate_dataset(X, Y, name, 5, 100)

#########################################  CERVICAL CANCER DATASET #########################################  
# definição do spirals
N<-1000 # numeros pares
p <- mlbench.spirals(N, 1, 0.15)
X <- p[[1]]
x1 <- X[,1]
x2 <- X[,2]
Y <- p[[2]]
name <- "Spirals Dataset"

evaluate_dataset(X, Y, name, 5, 100)

######################################### CAESARIAN DATASET #########################################  
name <- "Caesarian Dataset"
cae <- read.csv('caesarian.csv', header=TRUE) 
# pre processamento
cae$y <- as.factor(cae$y)

# converter em matrix
mat <- data.matrix(cae)
X <- scale(mat[, 1:5], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 6]

evaluate_dataset(X, Y, name, 4, 100)

#########################################  CERVICAL CANCER DATASET #########################################  
name <- "Breast Cancer Dataset"
bc.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
bc <- read.csv(bc.url, header=FALSE) 
bc <- bc[,2:11]
bc <- na.omit(bc)
bc <- droplevels(bc[!bc$V7 == '?',])
# pre processamento
bc$V11 <- as.factor(bc$V11)
names(bc)[10] <- "y"

# converter em matrix
mat <- data.matrix(bc)
X <- scale(mat[, 1:9], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 10]

evaluate_dataset(X, Y, name, 5, 100)

