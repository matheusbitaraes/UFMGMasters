# carregar 5 conjuntos de dados e transformá-los para binários (colocando saídas como -1 e 1).
library(mlbench)
# library(e1071)
library(caret)
library(kernlab)
library(GGClassification)


# spirals
N<-1000 #numeros pares
p <- mlbench.spirals(N,1,0.05)
x <- p[[1]]
x1 <- x[,1]
x2 <- x[,2]
y <- p[[2]]
spirals <- data.frame(x[,1], x[,2], y)
str(spirals)


inTraining <- createDataPartition(spirals$y, p = .80, list = FALSE)
training <- spirals[ inTraining,]
testing  <- spirals[-inTraining,]


# GABRIEL GRAPH
gg <- model(spirals, spirals$y)

# fitControl <- trainControl(## 10-fold CV
#   method = "svmLinear3",
#   number = 10,
#   ## repeated ten times
#   repeats = 10)


svmFit <- train(y ~ ., data = training, 
                 method = "svmRadial", 
                 verbose = TRUE)

svmFit
svmFit$finalModel
plot(svmFit$finalModel)

newlssvm <- getModelInfo()$lssvmLinear
newlssvm$fit <- function(x, y, wts, param, lev, last, classProbs, ...) {
  kernlab::lssvm(x = as.matrix(x), y = y,
                 tau = param$tau)    
}
svmls <- train(y ~ .,
               data = training,
               method = newlssvm,
               preProc = c("center", "scale")
)

plot(svmls$finalModel)

featurePlot(x = spirals, 
            y = spirals$y, 
            plot = "pairs",
            ## Add a key at the top
            auto.key = list(columns = 2))


#outra abordagem
model <- svm(y ~ ., data = spirals)
# alternatively the traditional interface:
# x <- subset(spirals, select = -y)
# y <- spirals$y
# model <- svm(x, y)
print(model)
summary(model)
# test with train data
pred <- predict(model, x)
# (same as:)
pred <- fitted(model)
# Check accuracy:
table(pred, y)
# compute decision values and probabilities:
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]

plot(model, spirals, fill=TRUE, symbolPalette = rainbow(3), color.palette = terrain.colors)


# WINE DATASET
wine.name <- "Wine Dataset"
wine.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine <- read.csv(wine.url, header=FALSE)
# pre-processamento
wine <- wine[1:130,]# filtra apenas dois tipos de vinho, para que fique binario
wine$V1[wine$V1 == 2] <- -1
wine$V1 <- as.factor(wine$V1)
names(wine)[1] <- 'y'

# BREAST CANCER DATASET
bc.name <- "Breast Cancer Dataset"
bc.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
bc <- read.csv(bc.url, header=FALSE) 
bc <- bc[,2:11]
bc$V11[bc$V11 == 2] <- -1
bc$V11[bc$V11 == 4] <- 1
# pre processamento
bc$V11 <- as.factor(bc$V11)
names(bc)[10] <- 'y'

# IRIS DATASET
iris.name <- "Iris Dataset"
iris.url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris <- read.csv(iris.url, header=FALSE)
# pre processamento
iris$V5[iris$V5 == 'Iris-setosa'] <- 1 #juntando classe 0 com classe 1
iris$V5[iris$V5 == 'Iris-versicolor'] <- 1 #juntando classe 0 com classe 1
iris$V5[iris$V5 == 'Iris-virginica'] <- -1 
iris$V5 <- as.factor(iris$V5) 
names(iris)[5] <- 'y'

# CAESARIAN DATASET
caesarian.name <- "Caesarian Dataset"
cae <- read.csv('caesarian.csv', header=TRUE) 
# pre processamento
cae$y[cae$y==0] <- -1
cae$y <- as.factor(cae$y)

# CERVICAL CANCER DATASET
cervcan.name <- "Cervical Cancer Dataset"
cervcan <- read.csv('sobar-72.csv', header=TRUE) 
# pre processamento
cervcan$ca_cervix[cervcan$ca_cervix==0] <- -1
cervcan$ca_cervix <- as.factor(cervcan$ca_cervix) 
names(cervcan)[20] <- 'y'

datasets = list(wine, bc, iris, cae, cervcan)
save(datasets, file="datasets.Rdata", compress=TRUE)


for (data in datasets){
  # realizar normalização dos dados
  print(data$y[1])
}



