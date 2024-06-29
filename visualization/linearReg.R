# Linear Regression height (cm) and bodymass(kg); source:https://www.theanalysisfactor.com/linear-models-r-plotting-regression-lines/

height <- c(176, 154, 138, 196, 132, 176, 181, 169, 150, 175)
bodymass <- c(82, 49, 53, 112, 47, 69, 77, 71, 62, 78)

# create simple plot
plot(bodymass, height)

# Enhance the plot
plot(bodymass, height, pch = 16, cex = 1.3, col = "blue", main = "HEIGHT PLOTTED AGAINST BODY MASS", xlab = "BODY MASS (kg)", ylab = "HEIGHT (cm)") # pch create solid dots, cex for dot size; main for title

# Perform Linear Regression
lm(height ~ bodymass) # lm->linear model; gives the intercept and the slope

# Plot the regression line
abline(lm(height ~ bodymass))


