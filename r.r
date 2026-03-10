# PRACTICAL 7
rainfall <- c(799,1174.8,865.1,1334.6,635.4,918.5,685.5,998.6,784.2,985,882.8,1071)
tsdata <- ts(rainfall, start=c(2012,1), frequency=12)
print(tsdata)
plot(tsdata)

# PRACTICAL 8
data <- iris[,1:4]
kc <- kmeans(data,3)
print(kc)
plot(data[,1:2], col=kc$cluster)
points(kc$centers[,1:2], col=1:3, pch=8, cex=2)

# PRACTICAL 9
x <- c(151,174,138,186,128,136,179,163,152,131)
y <- c(63,81,56,91,47,57,76,72,62,48)
model <- lm(y~x)
summary(model)
predict(model, data.frame(x=170))
plot(x,y)
abline(model)

# PRACTICAL 10
library(party)

data <- readingSkills[1:105,]
tree <- ctree(nativeSpeaker ~ age + shoeSize + score, data=data)
print(tree)
plot(tree)