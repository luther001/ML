dataset = read.csv(("Position_salaries.csv"))
dataset = dataset[2:3]


#Fitting Polynomial Regression to the dataset Set
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ .,data = dataset)

#visualizing the dataset
ggplot() + 
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') + 
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)), colour = 'blue') +
  ggtitle('R Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

y_pred = predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))