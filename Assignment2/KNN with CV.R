# Elliott Temkin
# GASF Data Science 2013
# Assignment 2
# Implement knn clustering algorithm with n-fold cross validation

# Load add-on packages
# install.packages("gridExtra", lib="Rpackages")
# Installation commented out for running script
library(ggplot2)
library(class)
library(gridExtra)

# Load data and separate labels
data <- iris
labels <- data$Species
data$Species <- NULL

# Establish variables for future use
# This is where the max n and k values are selected if you want to change them
NumRecs <- nrow(data)
err.rates <- data.frame()  
max.n <- 15
max.k <- 15
knn.fit <- factor(levels=c("setosa", "versicolor", "virginica"))

# Define KnnNfold function
KnnNfold <- function(n, set, cl, k, err.rates) {
  # Performs n-fold cross validation for a knn classifier
  #
  # Args:
  #   n: Number of folds in cross validation
  #   set: Data set to perform knn clustering on
  #   cl: Labels of classes associated with data in set
  #   k: The number of neighbors used in the knn classifier
  #
  # Returns:
  #   Generalization error
  #   Also prints confusion matrix for knn results v. correct labels
  
  # Randomize data and create index to cut into n pieces
  set.seed(1)
  NumRecs <- nrow(set)
  rand <- sample(1:NumRecs)
  set.rand <- set[rand, ]
  label <- cl[rand]
  train.divisions <- cut(1:NumRecs, n, labels = FALSE) 
  
  # Look through n different rounds for cross validation and run knn analysis
  for (NTest in 1:n) {
    test.index <- which(train.divisions == NTest)
    test.data <- set.rand[test.index, ]
    train.data <- set.rand[-test.index, ]
    test.labels <- as.factor(as.matrix(label)[test.index, ])
    train.labels <- as.factor(as.matrix(label)[-test.index, ])

    knn.fit[test.index] <- knn(train = train.data, 
	                           test  = test.data,
	                           cl    = train.labels,
	                           k     = k)
  }

  # Print out confusion matrix, calculate and store generalization error
  cat('\n', 'n = ', n, ', k = ', k, '\n', sep='')
  print(table(label, knn.fit))
  this.err <- sum(label != knn.fit) / length(label)
  return(this.err)
}

# Loop from k = 1 to max.k
for (k in 1:max.k) {

  # Loop from n = 2 to max.n and perform n-fold cross validation with KnnNfold function
  # Add error and k value to err.rates data frame
  for (n in 2:max.n) {
    err.rates[((k-1)*(max.n-1))+n-1, 1] <- KnnNfold(n, data, labels, k, err.rates)
    err.rates[((k-1)*(max.n-1))+n-1, 2] <- k    
  }
}

# Create results data frame and plot generalization error v. n for each k
# CV not particularly significant for knn, but good for general other algorithms
results <- data.frame(rep(2:max.n, max.k), err.rates)
names(results) <- c('n', 'err.rate', 'k')
results$k <- as.factor(results$k)
title <- paste('knn results')

results.plot <- ggplot(results, aes(x=n, y=err.rate)) + geom_line(aes(colour=k))
results.plot <- results.plot + ggtitle(title)
results$n <- as.factor(results$n)
results$k <- as.numeric(results$k)
results.plot2 <- ggplot(results, aes(x=k, y=err.rate)) + geom_line(aes(colour=n))
grid.arrange(results.plot, results.plot2)

