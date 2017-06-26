##install.packages("data.table",dependencies = T)
##install.packages("mlr",dependencies = T)


library(data.table)
library(mlr)
library(xgboost)


setwd("C:/Users/psrini1/Desktop/Pavitra/Others/XGboost")
getwd()


setcol <- c("age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","target")


train <- read.table("adult.data.txt",sep =",",header=F,col.names=setcol, na.strings=c(" ?"),stringsAsFactors = F)
test <- read.table("adult.test.txt",sep =",",header=F,col.names=setcol, na.strings=c(" ?"),stringsAsFactors = F,skip=1)

## In R, xgboost package uses a matrix of input data instead of a data frame.
## Convert data frame to data table

setDT(train)
setDT(test)

str(test)
str(train)


## Checking for missing values
table(is.na(train))
sapply(train, function(x) sum(is.na(x))/length(x))*100

table(is.na(test))
sapply(test, function(x) sum(is.na(x))/length(x))*100

##set all missing value as "Missing"
train[is.na(train)] <- "Missing"
test[is.na(test)] <- "Missing"


#quick data cleaning
#remove extra character / trailing blank space from target variable in test data
library(stringr)
test [,target := substr(target,start = 1,stop = nchar(target)-1)]

class(train)

## remove leading whitespace,since this is usally an issue related to string variables, pick only string type character variables

char_col <- colnames(train)[sapply(train,is.character)]

for(i in char_col) set(train,j=i,value = str_trim(train[[i]],side = "left"))

for(i in char_col) set(train,j=i,value = str_trim(train[[i]],side = "right"))

for(i in char_col) set(test,j=i,value = str_trim(test[[i]],side = "left"))

for(i in char_col) set(test,j=i,value = str_trim(test[[i]],side = "right"))

## Categorical variable to numeric matrix conversion using one hot encoding

new_tr <- model.matrix(~.+0, data = train[,-c("target"),with = F])

new_ts <- model.matrix(~.+0, data = test[,-c("target"),with = F])


## XGboost package accepts the target  variable separately and hence it is removed in the above step

## Converting the target variables into numeric datatype

labels <- as.factor(train$target)

ts_labels <- as.factor(test$target)

labels <- as.numeric(labels)-1   ## use -1 so that the levels start from 0

ts_labels <- as.numeric(ts_labels)-1

str(new_tr)

## Data table to matrix conversion (dense matrix conversion)

dtrain<- xgb.DMatrix(data=new_tr,label=labels)

dtest <- xgb.DMatrix(data = new_ts,label=ts_labels)

## with default parameters

default_params <- list(booster="gbtree",objective="binary:logistic",eta=0.3,gamma=0,max_depth=6,min_child_weight=1,subsample=1,colsample_bytree=1)

## Use xgb.cv to determine the optimal iteration count.Besides, xgb.CV also returns the test error

set.seed(200)

xgbcv <- xgb.cv(params=default_params,data=dtrain,nrounds=100,nfold=5,showsd=T,stratified = T,print_every_n = 10,early_stopping_round=20,maximize = F)

## First defualt model using iteration count from the previous step

xgb1 <- xgb.train(params = default_params,data=dtrain,nround =67,watchlist = list(val=dtest,train=dtrain),print.every.n=10,early_stopping_rounds = 10, maximize = F, eval_metric = "error" )

xgbpred <- predict(xgb1,dtest)

xgbpred <- ifelse (xgbpred > 0.5,1,0)


## Feature importance

xgb_importance <- xgb.importance(feature_names = colnames(new_tr),model=xgb1)

## Plotting the top 10 variables

xgb.plot.importance(xgb_importance[1:10])

op <- data.frame(cbind(test,xgbpred))

table(op$target,op$xgbpred)
