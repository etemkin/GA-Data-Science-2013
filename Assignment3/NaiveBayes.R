# Elliott Temkin
# GASF Data Science 2013
# Assignment 3
# Implement Naive Bayes classification with n-fold cross validation

# Load packages
# install.packages("e1071", lib = "Rpackages")
library("e1071")

# Define NaiveBayesNfold function
NaiveBayesNfold <- function(n, set, cl) {
  # Performs n-fold cross validation for a Naive Bayes classifier
  #
  # Args:
  #   n: Number of folds
  #   set: Data set to perform classification on
  #   cl: Labels of classes associated with data in set
  #
  # Returns:
  #   List giving inferred mean and standard deviation petal length distribution
  #   for three different species of iris along with generalization error
  #   Also prints confusion matrix for knn results v. correct labels
  
  # Randomize data and create index to cut into n pieces
  set.seed(1)
  NumRecs <- nrow(set)
  rand <- sample(1:NumRecs)
  set.rand <- set[rand, ]
  label <- cl[rand]
  train.divisions <- cut(1:NumRecs, n, labels = FALSE) 
  
  # Initialize vectors
  this.err <- vector()
  classifiers <- vector("list")
  setosa.pl.mean <- vector()
  setosa.pl.sd <- vector()
  versicolor.pl.mean <- vector()
  versicolor.pl.sd <- vector()
  virginica.pl.mean <- vector()
  virginica.pl.sd <- vector()
  
  # Look through n different rounds for cross validation and run Naive Bayes analysis
  for (NTest in 1:n) {
    test.index <- which(train.divisions == NTest)
    test.data <- set.rand[test.index, ]
    train.data <- set.rand[-test.index, ]
    test.labels <- as.factor(as.matrix(label)[test.index, ])
    train.labels <- as.factor(as.matrix(label)[-test.index, ])

    # Run Naive Bayes classifier for test set, compare with training, return outputs
    classifiers[[NTest]] <- naiveBayes(train.data, train.labels)	
    this.err[NTest] <- sum(predict(classifiers[[NTest]], test.data)
	                       != test.labels) / length(test.labels)
	setosa.pl.mean[NTest] <- classifiers[[NTest]]$tables$Petal.Length[1,1]
	setosa.pl.sd[NTest] <- classifiers[[NTest]]$tables$Petal.Length[1,2]
	versicolor.pl.mean[NTest] <- classifiers[[NTest]]$tables$Petal.Length[2,1]
	versicolor.pl.sd[NTest] <- classifiers[[NTest]]$tables$Petal.Length[2,2]
	virginica.pl.mean[NTest] <- classifiers[[NTest]]$tables$Petal.Length[3,1]
	virginica.pl.sd[NTest] <- classifiers[[NTest]]$tables$Petal.Length[3,2]	
  }

  # Calculate and store generalization error, and average of means, SD's for each fold
  # Note that SD's are determined by using the square root of the average variance
  error <- mean(this.err)
  setosa.pl.mean <- mean(setosa.pl.mean)
  setosa.pl.sd <- sqrt(mean(setosa.pl.sd^2))
  versicolor.pl.mean <- mean(versicolor.pl.mean)
  versicolor.pl.sd <- sqrt(mean(versicolor.pl.sd^2))
  virginica.pl.mean <- mean(virginica.pl.mean)
  virginica.pl.sd <- sqrt(mean(virginica.pl.sd^2))
  
  # Return results as a list
  results <- list("setosa.pl.mean"     = setosa.pl.mean,
                  "setosa.pl.sd"       = setosa.pl.sd,
                  "versicolor.pl.mean" = versicolor.pl.mean,
                  "versicolor.pl.sd"   = versicolor.pl.sd,
                  "virginica.pl.mean"  = virginica.pl.mean,
                  "virginica.pl.sd"    = virginica.pl.sd,
                  "error"              = error)
  return(results)
}

# Load data and separate
data <- iris
labels <- data$Species
data$Species <- NULL

# Define max n for cross validation and initialize more vectors
max.n <- 10
results <- array(list())
error <- vector()

# Sweep through n's, call NaiveBayesNfold function, and store results
for (n in 2:max.n) {
  results[[n-1]] <- NaiveBayesNfold(n, data, labels)
  error[n-1] <- results[[n-1]]$error
  cat('\n', 'n = ', n, '\n', sep='')
  cat('generalization error = ', error[n-1], '\n')
}

# Determine which n had the lowest generalization error
# In this case GE doesn't seem to be affected much by choice of n
# But, we'll go ahead and use min. GE to determine which results to plot
# Also note that this is the index we're returning -- the n is actually one greater
# i.e., in this case min.error is 6, corresponding to n=7 having the lowest GE
min.error <- which.min(error)

# Set up plot and print setosa petal length distribution curve
plot(function(x) dnorm(x,
                       results[[min.error]]$setosa.pl.mean, 
                       results[[min.error]]$setosa.pl.sd),
                       0,
                       8,
                       col  = "red",
                       main = "Petal length distribution for the 3 different species",
                       xlab = "Length of petal",
                       ylab = "Distribution",
                       sub = "red = setosa, blue = versicolor, green = virginica")

# Add versicolor curve
curve(dnorm(x,
            results[[min.error]]$versicolor.pl.mean, 
            results[[min.error]]$versicolor.pl.sd),
            add=TRUE,
            col="blue")

# Finally virginica curve            
curve(dnorm(x,
		    results[[min.error]]$virginica.pl.mean, 
            results[[min.error]]$virginica.pl.sd),
            add=TRUE,
            col="green")
            
# That's it.  We've run an n-fold cross validation on a Naive Bayes classifier for irises.
# Choice of n does not seem to have an appreciable affect on generalization error.
# I arbitrarily selected petal length as the feature to track and plot, though any of 
# the others could have been examined.  In this case, it appears petal length can clearly
# classify between setosa and the two other species.  However, versicolor and virginica
# appear to have overlapping petal length distributions, suggesting other factors would
# need to contribute to improve classification between these species.