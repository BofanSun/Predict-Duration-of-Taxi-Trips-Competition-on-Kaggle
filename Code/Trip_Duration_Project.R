library(lubridate)
library(tidyverse)
library(ggplot2)
library(glmnet)
library(plotmo)
library(geosphere)
library(randomForest)
library(caret)
library(ranger)
library(gbm)

rmsle <- function(pi, ai){
  index = which(ai!=0)
  term = (log(pi[index]+1) - log(ai[index]+1))^2
  result = sqrt(mean(term))
  return(result)
}

#read data
train_dat <- read.csv('Desktop/w22proj1/W22P1_train.csv', header = TRUE)
test_dat <- read.csv("Desktop/w22proj1/W22P1_test.csv", header = TRUE)

glimpse(train_dat)
summary(train_dat)

#predicts the total ride duration of taxi trips in New York City
#Data visualization
g1 = ggplot(data=train_dat, aes(x = id, y = trip_duration)) + geom_point()
g2 = ggplot(data=train_dat, aes(x = passenger_count, y = trip_duration)) + geom_point()
g3 = ggplot(data=train_dat, aes(x = pickup_longitude, y = trip_duration)) + geom_point()
g4 = ggplot(data=train_dat, aes(x = pickup_latitude, y = trip_duration)) + geom_point()
g5 = ggplot(data=train_dat, aes(x = dropoff_longitude, y = trip_duration)) + geom_point()
g6 = ggplot(data=train_dat, aes(x = dropoff_latitude, y = trip_duration)) + geom_point()
g7 = ggplot(data=train_dat, aes(x = pickup_longitude, y = pickup_latitude)) + geom_point()
g8 = ggplot(data=train_dat, aes(x = dropoff_longitude, y = dropoff_latitude)) + geom_point()
ggarrange(g1, g2, g3, g4, g5, g6, g7, g8)

g9 = ggplot(data=train_dat, aes(x=passenger_count))+geom_histogram(bins=20)
g10 = ggplot(data=train_dat,aes(x=trip_duration))+geom_histogram(bins=100)+coord_cartesian(x=c(0,90000))
ggarrange(g9, g10)

#Transforming data
#extract hour from pickup_time
train_dat$hour = hour(hms(train_dat$pickup_time))
test_dat$hour = hour(hms(test_dat$pickup_time))

#extract month and day from pickup_date
train_dat$pickup_day = day(ymd(train_dat$pickup_date))
test_dat$pickup_day = day(ymd(test_dat$pickup_date))
train_dat$pickup_wday = wday(ymd(train_dat$pickup_date))
test_dat$pickup_wday = wday(ymd(test_dat$pickup_date))

train_dat$weekend = ifelse((train_dat$pickup_wday == 6 | train_dat$pickup_wday == 7), 1, 0)
test_dat$weekend = ifelse((test_dat$pickup_wday == 6 | test_dat$pickup_wday == 7), 1, 0)

#transform longitude and latitude into distance
pickup<- as.matrix(data.frame(train_dat$pickup_longitude, train_dat$pickup_latitude))
dropoff<-as.matrix(data.frame(train_dat$dropoff_longitude, train_dat$dropoff_latitude))
train_dat$distance <- round(distHaversine(pickup, dropoff, r=6378137))

pickup_test<- as.matrix(data.frame(test_dat$pickup_longitude, test_dat$pickup_latitude))
dropoff_test<-as.matrix(data.frame(test_dat$dropoff_longitude, test_dat$dropoff_latitude))
test_dat$distance <- round(distHaversine(pickup_test, dropoff_test, r=6378137))

#Data visualization on the newly-added features
ggplot(data=train_dat,aes(x=distance))+geom_histogram(bins=100)+coord_cartesian(x=c(0,30000))
ggplot(data=train_dat, aes(x=pickup_wday))+geom_histogram(bins=20)
ggplot(data=train_dat, aes(x = pickup_day, y = trip_duration)) + geom_point()
ggplot(data=train_dat, aes(x = pickup_wday, y = trip_duration)) + geom_point()
ggplot(data=train_dat, aes(x = distance, y = trip_duration)) + geom_point()

#Modeling
#remove outliners
train_dat <- train_dat %>%
  filter(passenger_count >= 1)

#dataset for model 1
drops <- c('id', 'pickup_date', 'pickup_time', 'pickup_longitude', 
           'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude')
train <- train_dat[, !names(train_dat) %in% drops]
test <- test_dat[, !names(test_dat) %in% drops]
trip_Duration <- train_dat$trip_duration

n = nrow(train)
index = sample(1:n, round(0.3*n))
test_1 = train[index,]
train_1 = train[-index,]

#dataset for model 2 and 3
drops <- c('id', 'pickup_date', 'pickup_time')
train1 <- train_dat[, !names(train_dat) %in% drops]
test1 <- test_dat[, !names(test_dat) %in% drops]
trip_Duration <- train$trip_duration

n = nrow(train_dat)
index = sample(1:n, round(0.3*n))
test_2 = train1[index,]
train_2 = train1[-index,]

#Model1: ridge regression + lasso regression
#Lasso regression
mylasso <- glmnet(as.matrix(train_1[,-2]), log(train_1$trip_duration), alpha = 1)
plot_glmnet(mylasso, label = TRUE, xvar = "lambda")
lasso.cv.out <- cv.glmnet(as.matrix(train_1[,-2]), log(train_1$trip_duration), alpha = 1)
plot(lasso.cv.out)
#lambda one-standard deviation
Dtest.1se <- predict(lasso.cv.out, s = lasso.cv.out$lambda.1se, newx=as.matrix(test_1[,-2]))
rmsle(exp(Dtest.1se), trip_Duration[index])
#[1] 0.6316224
#lambda smallest cross validation error
lass.min <- predict(lasso.cv.out, s = lasso.cv.out$lambda.min, newx = as.matrix(test_1[,-2]))
rmsle(exp(lass.min), trip_Duration[index])
#[1] 0.6229381

#ridge regression
myridge <- glmnet(as.matrix(train_1[-2]), log(train_1$trip_duration), alpha = 0)
plot_glmnet(myridge, label = TRUE, xvar = "lambda")
ridge.cv.out <- cv.glmnet(as.matrix(train_1[-2]), log(train_1$trip_duration), alpha = 0)
plot(ridge.cv.out)
#lambda one-standard deviation
ridge.1se <- predict(ridge.cv.out, s = ridge.cv.out$lambda.1se, newx=as.matrix(test_1[-2]))
rmsle(exp(ridge.1se), trip_Duration[index])
#[1] 0.6342622
#lambda smallest cross validation error
ridge.min <- predict(ridge.cv.out, s = ridge.cv.out$lambda.min, newx = as.matrix(test_1[-2]))
rmsle(exp(ridge.min), trip_Duration[index])
#[1] 0.6244022

#combine lasso and ridge regression together
a=0.1
b=0.4
c=0.1
d=0.4
combine <- Dtest.1se*a+lass.min*b+ridge.1se*c+ridge.min*d
rmsle(exp(combine), trip_Duration[index])
#[1] 0.6244389
lasso0 <- predict(lasso.cv.out, s = lasso.cv.out$lambda.1se, newx = as.matrix(test))
lasso1<- predict(lasso.cv.out, s = lasso.cv.out$lambda.min, newx = as.matrix(test))
ridge0 <- predict(ridge.cv.out, s = ridge.cv.out$lambda.1se, newx=as.matrix(test))
ridge1 <- predict(ridge.cv.out, s = ridge.cv.out$lambda.min, newx=as.matrix(test))
combine0 <- lasso0*a+lasso1*b+ridge0*c+ridge1*d
out_Dat = data.frame(id = test_dat$id, trip_duration = exp(combine0))
colnames(out_Dat) = c('ID', 'trip_duration')
write.csv(out_Dat, "Desktop/w22proj1/W22P1_submission.csv", row.names = FALSE)


#Model 2: Random Forest
#choose the best tunning parameter mtry
optmtry = tuneRF(train_2[-6], log(train_2$trip_duration), stepFactor = 2, plot = TRUE)

rfModel <- ranger(log(trip_duration)~., data = train_2, importance="impurity", 
                  num.trees = 500, mtry = 3)
rf.pred = predict(rfModel, test_2[,-6])
rmsle(exp(rf.pred$predictions), trip_Duration[index])
#[1] 0.4374243

rf_pred <- predict(rfModel, as.matrix(test1))
out_Dat_2 = data.frame(id = test_dat$id, trip_duration = exp(rf_pred$predictions))
colnames(out_Dat_2) = c('id', 'trip_duration')
write.csv(out_Dat_2, "Desktop/w22proj1/Result_rf1.csv", row.names = FALSE)


#Model 3: boosting
#select key parameters
ctrl <- trainControl(method = "repeatedcv",
                     number = 10,
                     repeats = 5)

gbm.Grid = expand.grid(interaction.depth = c(2,3,4,5,6), 
                       n.trees = (1:5)*200, 
                       shrinkage = c(0.15, 0.1, 0.05),
                       n.minobsinnode = 10) 
gbm.cv.model <- train(log(trip_duration)~., data=train_3, method = "gbm", trControl = ctrl,
                      tuneGrid = gbm.Grid, verbose = FALSE)
gbm.cv.model
cv.num = gbm.perf(boost.trip)
boost.pred=exp(redict(gbm.cv.model,newdata=test_2[-6], n.trees = cv.num))
rmsle(boost.pred, trip_Duration[index])

boosting_pred <- predict(gbm.cv.model, as.matrix(test1), n.trees = cv.num)
out_Dat_3 = data.frame(id = test_dat$id, trip_duration = exp(boosting_pred))
colnames(out_Dat_3) = c('id', 'trip_duration')
write.csv(out_Dat_3, "Desktop/w22proj1/W22P1_submission_boosting.csv", row.names = FALSE)




