# Elliott Temkin
# Introduction to Data Science
# Assignment 1

library(ggplot2)
x = read.table('http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data')	
																					# Load data into table
head(x) 																			# Confirm proper loading
headers = c("mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model year", "origin", "car name")
																					# Create names	
names(x) = headers																	# Apply names to data frame
# str(x)																			# Look at data's structure
plot(x)																				# Look at correlation

linear.fit = lm(mpg ~ weight, data=x)												# Fit linear model for weight and MPG
lmsum = summary(linear.fit)															# Create summary of linear model
print(lmsum)																		# Print summary

weightsquared = x$weight^2															# Create weightsquared vector
x[10] = weightsquared																# Add vector to data frame
names(x)[10]= "weightsquared"														# Name vector
square.fit = lm(mpg ~ weight+weightsquared, data=x)									# Fit for quadratic
squaresum=summary(square.fit)														# Create summary
print(squaresum)																	# Print summary

# Increase in R-squared and adjusted R-squared

weightcubed = x$weight^3															# Same as above, but weightcubed added
x[11] = weightcubed
names(x)[11]= "weightcubed"
cube.fit = lm(mpg ~ weight+weightsquared+weightcubed, data=x)
cubesum=summary(cube.fit)
print(cubesum)

# Same R-squared but slight decrease in adjusted R-squared and increase in RSE - overfitting

# install.packages("MASS", lib="Rpackages") 										# Install MASS package
# library(MASS)																		
# help(lm.ridge)																	# Read about lm.ridge
ridge = lm.ridge(mpg ~ weight+weightsquared, data = x)								# Run ridge regression
print(ridge)																		# Similar coefficients