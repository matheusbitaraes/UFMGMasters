library(mlbench)
library(kernlab)
library(GGClassification)
library(caret)
library(spatgraphs)

# definição do spirals
N<-1000 #numeros pares
p <- mlbench.spirals(N,1,0.05)
x <- p[[1]]
x1 <- x[,1]
x2 <- x[,2]
y <- p[[2]]
spirals <- data.frame(x[,1], x[,2], y)
str(spirals)

# divisão em treinamento e teste
inTraining <- createDataPartition(spirals$y, p = .8, list = FALSE)
training <- spirals[ inTraining,]
testing  <- spirals[-inTraining,]


# SVM
svmtrein <- kernlab::ksvm(y ~ ., data = training, type='C-svc', kernel='rbfdot')
svmtrein
predSVM <- kernlab::predict(svmtrein, testing, type='response')
table(testing$y, predSVM)
kernlab::plot(svmtrein, data=spirals)

# LSSVM
lssvmtrein <- kernlab::ksvm(y ~ ., data = training, type='C-svc', kernel='rbfdot')
lssvmtrein
predLSSVM <- kernlab::predict(lssvmtrein, testing, type='response')
table(testing$y, predLSSVM)
kernlab::plot(lssvmtrein, data=spirals)

# GABRIEL GRAPH
a = x[y==1,]
b = x[y==2,]
g <- spatgraph(x, type="gabriel")
mdl <- GGClassification::model(x, y)
mdl
prd <- predict(mdl, x)
prd

# https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/points.html
plot(g, x)
points(a, col="blue", pch=1)
points(b, col="green", pch=1)
points(mdl$Midpoints, col="red", pch=19)
title(main="Grafo de Gabriel")


#dataset 2
nc = 100
xc1 <- matrix(0.3 * rnorm(nc) + 2.5, ncol = 2)
xc2 <- matrix(0.3 * rnorm(nc) + 3.5, ncol = 2)
xc1 <- cbind(xc1, rep(0, times = nc/2))
xc2 <- cbind(xc2, rep(1, times = nc/2))
X <- rbind(xc1, xc2)
suffled_indexes <- sample(nc)
train_size = nc * 0.7
X_train <- X[suffled_indexes[1:train_size], cbind(1,2)]
y_train <- X[suffled_indexes[1:train_size], 3]
X_test <- X[suffled_indexes[(71:100)], cbind(1,2)]
y_test <- X[suffled_indexes[(71:100)], 3]


# SVM
svmtrein <- kernlab::ksvm(X[,1:2], X[,3], type='C-svc', kernel='rbfdot')
svmtrein
predSVM <- kernlab::predict(svmtrein, X_test, type='response')
table(y_test, predSVM)
kernlab::plot(svmtrein, data=X)

# LSSVM
lssvmtrein <- kernlab::ksvm(X[,1:2], X[,3], data = X_train, type='C-svc', kernel='rbfdot')
lssvmtrein
predLSSVM <- kernlab::predict(lssvmtrein, X_test, type='response')
table(y_test, predLSSVM)
kernlab::plot(lssvmtrein, data=X, main="t", xlab="oi")


a = X[X[,3] == 1,1:2]
b = X[X[,3] == 0,1:2]
g <- spatgraph(X[,1:2], type="gabriel")
mdl <- GGClassification::model(X_train, y_train)
mdl
prd <- predict(mdl, X_test)
prd

# https://stat.ethz.ch/R-manual/R-devel/library/graphics/html/points.html
plot(g, X[,1:2])
points(a, col="blue", pch=1)
points(b, col="green", pch=1)
points(mdl$Midpoints, col="red", pch=19)
title(main="Grafo de Gabriel")
