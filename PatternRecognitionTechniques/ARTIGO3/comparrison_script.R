# carregar 5 conjuntos de dados e transformá-los para binários (colocando saídas como -1 e 1).
library(mlbench)
library(e1071)
library(caret)
library(kernlab)
library(GGClassification)
library(tsutils)
library(kerndwd)

classify <- function(data_, name, X, Y, svmKernel, lssvmKernel) {
  
  #data_ <- bc
  test_perc <- 0.3
  N = 50 # numero de execuções para calculo de acurácia
  
  print(name)
  control <- trainControl(method="repeatedcv", number=10, repeats=4)
  print("trainning svm...")
  trainSVM <- train(y ~ ., data=data_, method=svmKernel, trControl=control, sigma=1)
  print("trainning lssvm...")
  trainLSSVM <- train(y ~ ., data=data_, method=lssvmKernel, trControl=control)
  print("done")
  modelSVM <- trainSVM$finalModel
  modelLSSVM <-trainLSSVM$finalModel
  modelGabriel <- GGClassification::model(X, Y)
  
  print("SVM:")
  print(trainSVM)
  
  print("LSSVM:")
  print(trainLSSVM)
  
  print("GG:")
  print(modelGabriel)
  
  accs <- data.frame() 
  kappas <- data.frame()
  for (i in 1:N){
    testIndexes <- createDataPartition(Y, p = test_perc, list = FALSE)
    x_test <- X[ testIndexes,]
    y_test <- Y[ testIndexes]
    data_test <- data_[ testIndexes,]
    y_pred_svm <- kernlab::predict(modelSVM, data_test[,names(data_test) != "y"], type='response')
    y_pred_lssvm <- kernlab::predict(modelLSSVM, data_test[,names(data_test) != "y"], type='response')
    y_pred_gg <- GGClassification::predict(modelGabriel, x_test)
    
    cm_svm <- confusionMatrix(y_pred_svm, data_test$y)
    cm_lssvm <- confusionMatrix(y_pred_lssvm, data_test$y)
    cm_gg <- confusionMatrix(as.factor(y_pred_gg), as.factor(y_test))
    SMV_acc <- cm_svm$overall["Accuracy"]
    LSSVM_acc <- cm_lssvm$overall["Accuracy"]
    GG_acc <- cm_gg$overall["Accuracy"]
    SMV_kappa <- cm_svm$overall["Kappa"]
    LSSVM_kappa <- cm_lssvm$overall["Kappa"]
    GG_kappa <- cm_gg$overall["Kappa"]
    accs_row = data.frame(SMV_acc, LSSVM_acc, GG_acc, row.names = i)
    kappas_row = data.frame(SMV_kappa, LSSVM_kappa, GG_kappa, row.names = i)
    accs <- rbind(accs, accs_row)
    kappas <- rbind(kappas, kappas_row)
  }
  
  print("RESULTS")
  print(summary(accs))
  print(summary(kappas))
  
  # Boxplots
  par(mar=c(4,10,1,1))
  boxplot(accs, data=data,
          las = 2,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = "Boxplots das Acurácias")
  par(mar=c(4,10,1,1))
  boxplot(kappas, data=data,
          las = 2,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = "Boxplots dos kappas")
  
  #aplicação do kruskal e teste de nemenyi
  accs_matrix <- as.matrix(accs)
  kappa_matrix <- as.matrix(kappas)
  tsutils::nemenyi(accs_matrix, conf.level=0.95,plottype="vmcb", main=sprintf("Accuracy Nemenyi test for %s", name)) # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"
  tsutils::nemenyi(kappa_matrix, conf.level=0.95,plottype="vmcb", main=sprintf("Kappa Nemenyi test for %s", name)) # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"
}

######################################### IRIS DATASET ######################################### 
iris.name <- "Iris Dataset"
iris.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris <- read.csv(iris.url, header=FALSE)
# pre processamento
iris$V5[iris$V5 == 'Iris-setosa'] <- 'Iris-versicolor' #juntando classe 0 com classe 1
iris$V5 <- factor(iris$V5)
iris$V5 <- as.factor(iris$V5)
names(iris)[5] <- 'y'
plot(iris)

# converter em matrix
mat <- data.matrix(iris)
X <- scale(mat[, 1:4], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 5]

# escolha de kernel
svmKernel <- 'svmRadial'
lssvmKernel <- 'lssvmRadial'

# chamar a função
classify(iris, iris.name, X, Y, svmKernel, lssvmKernel)

######################################### WINE DATASET ######################################### 
wine.name <- "Wine Dataset"
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

# escolha de kernel
svmKernel <- 'svmLinear'
lssvmKernel <- getModelInfo()$lssvmLinear
lssvmKernel$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  kernlab::lssvm(x = as.matrix(x), y = y,
                 tau = param$tau)    
}

# chamar a função
classify(wine, wine.name, X, Y, svmKernel, lssvmKernel)


######################################### PimaIndiansDiabetes ######################################### 
# load the dataset
pima <- data(PimaIndiansDiabetes)
names(PimaIndiansDiabetes)[9] <- "y"

# converter em matrix
mat <- data.matrix(PimaIndiansDiabetes)
X <- scale(mat[, 1:8], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 9]

# escolha de kernel
svmKernel <- 'svmLinear'
lssvmKernel <- getModelInfo()$lssvmLinear
lssvmKernel$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  kernlab::lssvm(x = as.matrix(x), y = y,
                 tau = param$tau)    
}

# chamar a função
classify(PimaIndiansDiabetes, "Pima Indians Diabetes", X, Y, svmKernel, lssvmKernel)

######################################### CAESARIAN DATASET #########################################  
caesarian.name <- "Caesarian Dataset"
cae <- read.csv('caesarian.csv', header=TRUE) 
# pre processamento
cae$y <- as.factor(cae$y)

# converter em matrix
mat <- data.matrix(cae)
X <- scale(mat[, 1:5], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 6]

# escolha de kernel
svmKernel <- 'svmLinear'
lssvmKernel <- getModelInfo()$lssvmLinear
lssvmKernel$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  kernlab::lssvm(x = as.matrix(x), y = y,
                 tau = param$tau)    
}

# chamar a função
classify(cae, caesarian.name, X, Y, svmKernel, lssvmKernel)

#########################################  CERVICAL CANCER DATASET #########################################  
cervcan.name <- "Cervical Cancer Dataset"
cervcan <- read.csv('sobar-72.csv', header=TRUE) 
# pre processamento
cervcan$ca_cervix[cervcan$ca_cervix==0] <- -1
cervcan$ca_cervix <- as.factor(cervcan$ca_cervix) 
names(cervcan)[20] <- 'y'

# converter em matrix
mat <- data.matrix(cervcan)
X <- scale(mat[, 1:19], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- mat[, 20]

# escolha de kernel
svmKernel <- 'svmLinear'
lssvmKernel <- getModelInfo()$lssvmLinear
lssvmKernel$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  kernlab::lssvm(x = as.matrix(x), y = y,
                 tau = param$tau)    
}

# chamar a função
classify(cervcan, cervcan.name, X, Y, svmKernel, lssvmKernel)




#########################################  BREAST CANCER DATASET - nope ######################################### 
bc.name <- "Breast Cancer Dataset"
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

# escolha de kernel
svmKernel <- 'svmLinear'
lssvmKernel <- getModelInfo()$lssvmLinear
lssvmKernel$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  kernlab::lssvm(x = as.matrix(x), y = y,
                 tau = param$tau)    
}

# chamar a função
classify(bc, bc.name, X, Y, svmKernel, lssvmKernel)


#########################################  BUPA LIVER DISORDER DATASET - nope ######################################### 
data(BUPA)

bupa <- data.frame(BUPA$X)
bupa$y <- BUPA$y

# converter em matrix
X <- scale(BUPA$X, center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification
Y <- BUPA$y

# escolha de kernel
svmKernel <- 'svmLinear'
lssvmKernel <- 'lssvmLinear'
lssvmKernel <- getModelInfo()$lssvmLinear
lssvmKernel$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  kernlab::lssvm(x = as.matrix(x), y = y,
                 tau = param$tau)    
}

# chamar a função
classify(bupa, "Bupa Liver Disorder Dataset", X, Y, svmKernel, lssvmKernel)


