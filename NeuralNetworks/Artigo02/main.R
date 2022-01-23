rm(list=ls())
graphics.off()

source('lda.R')
source('elm.R')
library(caret) 
library(mlbench)
library(clusterCrit)

# escolher os metodos de avaliação de partição do ClusterCrit do R
# estudar o conjunto de dados e aplicar os metodos de avaliação de qualidade de cluster
# selecionar variaveis ue vai melhorar o indice de qualidade do cluster

# clusterizar as features de acordo com regras x, y e z e depois avalia qual metrica ficou melhor, pelo metodo de avaliação de clusteres.
# x = kmeans
# y = ?
# z = ?

# aí joga no ELM e RBF para ver a qualidade. E compara com o que se acha na literatura

# LDA para avaliar os clusteres tbm?

# 1) pega todas as features
# 2) escolhe uns 4 indices de qualidade cluster e faz uma otimização das features que melhoram este indice
# 3) avalia os 4 conjuntos de features + todas as features para elm e LDA (ou outro) e ve qual prospera mais

feature_reduction <- function (X, Y, index_criterium) {
  max_feature_reduction_rate <- 0.5 # if there are 10 features, max reduction will be 10*rate features
  num_features <- ncol(X)
  min_features <- round(num_features * max_feature_reduction_rate)
  transp_x <- t(X)
  
  # for (index_criterium in index_criteria){
  vals <- vector()
  feature_length_array <- min_features:num_features-1
  for (k in feature_length_array){
    # Perform the kmeans algorithm
    cl <- kmeans(transp_x, k)
    
    # Compute the Calinski_Harabasz index
    vals <- c(vals, as.numeric(intCriteria(transp_x, cl$cluster,index_criterium)))
  }
  idx <- bestCriterion(vals,index_criterium)
  
  # faz o clustering com o numero indicado e deixa uma feature por cluster
  cl <- kmeans(transp_x, feature_length_array[idx])
  
  final_transposed_x = matrix(0, nrow=feature_length_array[idx], ncol=ncol(transp_x))
  j <- 1
  for (i in 1:feature_length_array[idx]) {
    
    clustered_features <- transp_x[cl$cluster == i,]
    num_cluster_features <- nrow(clustered_features)
    if (!is.null(num_cluster_features) && num_cluster_features > 1) { 
      # select only one feature from cluster
      remaining_row <- sample(num_cluster_features)[1]
      final_transposed_x[j,] <- clustered_features[remaining_row,]
    } else {
      final_transposed_x[j,] <- t(clustered_features)
    }
    j <- j + 1
  }
  
  mod_x <- t(final_transposed_x)
  
  return(mod_x)
}

# MAIN
multiple_evaluations <- function(X, Y, num_eval){
  
  index_criteria <- c('Calinski_Harabasz',
                      'Davies_Bouldin', 
                      'Banfeld_Raftery',
                      'McClain_Rao',
                      'GDI53',
                      'PBM',
                      'Point_Biserial')
  num_criteria <- length(index_criteria)
  
  Xmod <- list()
  for (j in 1:num_criteria){
    Xmod[[j]] <- feature_reduction(X, Y, index_criteria[j])
  }
  
  m1_acc <- matrix(0, nrow=num_eval, ncol=1+num_criteria)
  m1_time <- matrix(0, nrow=num_eval, ncol=1+num_criteria)
  m2_acc <- matrix(0, nrow=num_eval, ncol=1+num_criteria)
  m2_time <- matrix(0, nrow=num_eval, ncol=1+num_criteria)
  for (i in 1:num_eval){
    # print(sprintf("Iteration - %s", i))
    
    m1_acc[i,1] <- eval_elm(X, Y, should_plot_matrix=FALSE, i)[1]
    m2_acc[i,1] <- eval_lda(X, Y, should_plot_matrix=FALSE, i)[1]
    
    for (j in 1:num_criteria){
      m1_acc[i,j+1] <- eval_elm(Xmod[[j]], Y, should_plot_matrix=FALSE, i)[1]
      m2_acc[i,j+1] <- eval_lda(Xmod[[j]], Y, should_plot_matrix=FALSE, i)[1]
    }

    # print(sprintf("LDA: %s | ELM: %s", m1_acc, m2_acc))
  }
  
  
  accs <- data.frame("ELM" = m1_acc[,1],
                     "LDA" = m2_acc[,1])
  names(accs)[1] = paste("ELM", ncol(X), "fts")
  names(accs)[2] = paste("LDA", ncol(X), "fts")
                     
  for (j in 1:length(index_criteria)) {
    num_features <- ncol(Xmod[[j]])
    accs[paste("ELM", index_criteria[j], num_features, "fts")] <- m1_acc[,j]
    accs[paste("LDA", index_criteria[j], num_features, "fts")] <- m2_acc[,j]
    }
  return(list(accs))
}

evaluate_dataset <- function(X, Y, name, num_exec){
  results <- multiple_evaluations(X, Y, num_exec)
  accs <- results[[1]]
  
  print(name)
  n <- names(accs)
  accs_matrix <- as.matrix(accs)
  for (i in 1:ncol(accs_matrix)){
    avg <- round(mean(accs_matrix[,i]), digits=4)
    std <- round(sd(accs_matrix[,i]), digits=4)
    print(sprintf("%s: %s +- %s", n[i], avg, std))
  }
    
  # par(mfrow=c(1,2))
  
  # Boxplot
  par(mar=c(2,16,1,1))
  boxplot(accs, data=data,
          las = 1,
          horizontal = TRUE,
          ann=FALSE
  )
  title(main = sprintf("Acurácias (%s)",name))
  
  # Aplicação do kruskal e teste de nemenyi
  accs_matrix <- - as.matrix(accs)
  tsutils::nemenyi(accs_matrix, conf.level=0.95,plottype="vmcb", sort=TRUE, main=sprintf("Teste de Nemenyi Acurácias (%s)", name)) # possiveis plots: "vline", "none", "mcb", "vmcb", "line", "matrix"
  }


######################################### IONOSPHERE DATASET ######################################### 
name <- "Ionosphere Dataset"
data(ionosphere, package = 'KernelKnn')
x_class = ionosphere[, -c(2, ncol(ionosphere))]
X = as.matrix(x_class)
dimnames(X) = NULL
Y = as.numeric(ionosphere[, ncol(ionosphere)]) - 1

evaluate_dataset(X, Y, name, 50)


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
Y <- mat[, 1] - 1 # transformando em 0 e 1

evaluate_dataset(X, Y, name, 50)


######################################### PimaIndiansDiabetes ######################################### 
# load the dataset
name <- "Pima Indians"
pima <- data(PimaIndiansDiabetes)
names(PimaIndiansDiabetes)[9] <- "y"

# converter em matrix
mat <- data.matrix(PimaIndiansDiabetes)
Y <- mat[, 9] - 1 # transformando entradas em 0 e 1
X <- scale(mat[, 0:8], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification

evaluate_dataset(X, Y, name, 50)


#########################################  CERVICAL CANCER DATASET #########################################  
name <- "Cervical Cancer Dataset"
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
Y <- mat[, 10] - 1 # subtraindo para ficar entre 0 e 1
X <- scale(mat[, 1:9], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification


evaluate_dataset(X, Y, name, 50)

#########################################  SONAR DATASET #########################################  
# load the dataset
name <- "Sonar Dataset"
sonar <- data(Sonar)

# converter em matrix
mat <- data.matrix(Sonar)
Y <- mat[, 61] - 1 # transformando entradas em 0 e 1
X <- scale(mat[, 0:60], center = TRUE, scale = TRUE) # fase de normalização dos dados para o gg classification

evaluate_dataset(X, Y, name, 50)

