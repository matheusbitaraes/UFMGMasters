rm(list=ls())
graphics.off()

library(e1071)
library(caret)
library(kernlab)
library(GGClassification)

# Carregar datasets
nc = 500
xc1 <- matrix(0.3 * rnorm(nc) + 2.5, ncol = 2)
xc2 <- matrix(0.3 * rnorm(nc) + 3.5, ncol = 2)
xc1 <- cbind(xc1, rep(0, times = nc/2))
xc2 <- cbind(xc2, rep(1, times = nc/2))
X <- rbind(xc1, xc2)
nc <- nrow(X)
suffled_indexes <- sample(nc)
train_size <- floor(nc * 0.8)
x_train <- X[suffled_indexes[1:train_size], cbind(1,2)]
y_train <- X[suffled_indexes[1:train_size], 3]
x_test <- X[suffled_indexes[(train_size+1):nc], cbind(1,2)]
y_test <- X[suffled_indexes[(train_size+1):nc], 3]
plot(xc1[,1], xc1[,2], col="blue", xlim=c(1, 5), ylim=c(1, 5)) # plot do dataset
points(xc2[,1], xc2[,2], col="red", pch=1) # plot do dataset
title(main="Two Gaussians Dataset with clusters")

# clusterizar o dataset
nclusters <- 5
km <- kmeans(as.matrix(x_train[,1:2]), nclusters, iter.max=50)
clusters <- as.matrix(km[[1]])

# plot cluster centers
points(km$centers, col="green", pch=16)

# plot samples
pchs <- c()
names <- c()
for (i in 1:nclusters){
  pch = 5 + i
  points(x_train[clusters == i,1], x_train[clusters == i,2], pch=pch)
  pchs <- c(pchs, pch)
  names <- c(names, sprintf("Cluster %s", i))
}
legend("topleft", legend=names, pch=pchs)


# treinamento de svm em clusteres
svms <- c()
for (i in 1:nclusters){
  print(sprintf("treinando cluster %s", i))
  x_ <-x_train[clusters == i,]
  y_ <-y_train[clusters == i]
  
  if (mean(y_)==1){
    svmtrein <- 1
  }
  else if (mean(y_)==0) {
    svmtrein <- 0
  } else {
    svmtrein <- kernlab::ksvm(x_, y_, type='C-svc', kernel='rbfdot')
    # svmtrein <- GGClassification::model(x_, y_)
  }
  svms <- c(svms, svmtrein)
  
}
p <- kernlab::predict(svmtrein, x_test)
p <- kernlab::predict(svms[[5]], x_test)


# x_points <- c(2.5, 3) 
# euc.dist <- function(x1, x2) sqrt(sum((x1 - x2) ^ 2))

get_cluster <- function(x, km){
  dists <- rowSums(x - km$centers)^2 
  cluster_id <- which(dists==min(dists))
  return (cluster_id)
}

pred_func <- function (x, svm_list, km){
  x <- cbind(x, 0)
  
  # atribui clusters
  for (i in 1:nrow(x)){
    cluster_id <- get_cluster(x[i,], km)
    x[i,3] <- cluster_id
  }
  
  x <- cbind(x, 0)
  for (cid in 1:nclusters){
    svm <- svm_list[[cid]]
    if (is.numeric(svm)){ # checa se é um numero
      x[x[,3]==cid, 4] <- svm
    }else{
      if (nrow(x[x[,3]==cid,])>0){
        x[x[,3]==cid, 4] <- kernlab::predict(svm, x[x[,3]==cid,1:2],
                                             type='response')
      }
    }
  }
  return(x[,4])
}

y_pred <- pred_func(x=x_test, svm_list=svms, km=km)
cm_nm <- confusionMatrix(as.factor(y_pred), as.factor(y_test))


# SVM
svmtrein <- kernlab::ksvm(x_train, y_train, type='C-svc', kernel='rbfdot')
predSVM <- kernlab::predict(svmtrein, x_test, type='response')
cm_svm <- confusionMatrix(as.factor(predSVM), as.factor(y_test))
# kernlab::plot(svmtrein, data=X)

#GG
mdl <- GGClassification::model(x_train, y_train)
y_pred_gg <- predict(mdl, x_test)
cm_gg <- confusionMatrix(as.factor(y_pred_gg), as.factor(y_test))


print("EVALUATION")
print(sprintf("cluster model: %s ",cm_nm$overall["Accuracy"]))
print(sprintf("SVM: %s ", cm_svm$overall["Accuracy"]))
print(sprintf("ggClassification: %s ",cm_gg$overall["Accuracy"]))
