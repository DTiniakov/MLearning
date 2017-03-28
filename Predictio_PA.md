# prediction PA
Daniil Tiniakov  
## Loading packages

```r
suppressMessages(library(caret))
suppressMessages(library(randomForest))
suppressMessages(library(rpart))
suppressMessages(library(rattle))
suppressMessages(library(gbm))
```

## Data processing
# Loading and reading data

```r
download.file(url='https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',destfile = 'training.csv')
download.file(url='https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv',destfile='testing.csv')
training<-read.csv('training.csv',na.strings=c("NA","#DIV/0!",""))[,-(1:2)]
testing<-read.csv('testing.csv',na.strings=c("NA","#DIV/0!",""))[,-(1:2)]
# Without row index and username vars
```

# Creating cross-validation set

```r
set.seed(2445475)
inTrain<-createDataPartition(y=training$classe, p=.5, list=FALSE)
train.set<-training[inTrain,]
cv.set<-training[-inTrain,]
dim(train.set)
```

```
## [1] 9812  158
```
# Deleting vars with near0 variance prediction and with lot of NAs


```r
percent.NA<-colSums(is.na(train.set))/nrow(train.set)
lots.NA<-which(percent.NA>.6)
train.set<-train.set[,-lots.NA]
train.set<-train.set[,-nearZeroVar(train.set)]
train.set<-train.set[,-3]
dim(train.set)
```

```
## [1] 9812   56
```

## Building models
After some experiments with RAM i decided to stop on decision trees, random forest and LDA models

```r
m1<-train(classe~.,data=train.set,method='rpart')
m4<-randomForest(classe~.,data=train.set,type="class")
m6<-train(classe~.,data=train.set,method='lda')
```

```
## Loading required package: MASS
```
From confusion table lowest insample error is shown by RF model (100% accuracy),LDA model shows 85% accuracy, and decision tree model only 57% accuracy.

```r
pred1<-predict(m1,newdata=train.set)
pred4<-predict(m4,newdata=train.set)
pred6<-predict(m6,newdata=train.set)
confusionMatrix(pred1,train.set$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2531  810  799  737  258
##          B   38  646   50  282  214
##          C  216  443  862  589  493
##          D    0    0    0    0    0
##          E    5    0    0    0  839
```

```r
confusionMatrix(pred4,train.set$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2790    0    0    0    0
##          B    0 1899    0    0    0
##          C    0    0 1711    0    0
##          D    0    0    0 1608    0
##          E    0    0    0    0 1804
```

```r
confusionMatrix(pred6,train.set$classe)$table
```

```
##           Reference
## Prediction    A    B    C    D    E
##          A 2337  263  171   87   82
##          B   75 1246  173   73  235
##          C  165  228 1122  186  153
##          D  204   88  208 1213  183
##          E    9   74   37   49 1151
```

## Cross-validation (Out of sample error)
Being aware of overfitting i decided to test models on cross-validation set.


```r
cv4<-predict(m4,newdata=cv.set)


confusionMatrix(cv4,cv.set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2790   13    0    0    0
##          B    0 1885    9    0    0
##          C    0    0 1702   11    0
##          D    0    0    0 1597    6
##          E    0    0    0    0 1797
## 
## Overall Statistics
##                                           
##                Accuracy : 0.996           
##                  95% CI : (0.9946, 0.9972)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.995           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9932   0.9947   0.9932   0.9967
## Specificity            0.9981   0.9989   0.9986   0.9993   1.0000
## Pos Pred Value         0.9954   0.9952   0.9936   0.9963   1.0000
## Neg Pred Value         1.0000   0.9984   0.9989   0.9987   0.9993
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2844   0.1922   0.1735   0.1628   0.1832
## Detection Prevalence   0.2857   0.1931   0.1746   0.1634   0.1832
## Balanced Accuracy      0.9991   0.9960   0.9967   0.9962   0.9983
```
Still RF model performs better (99.5% accuracy, Kappa values =.995)


## Predictions using chosen RF model

```r
predict.test<-predict(m4,newdata=testing)
predict.test
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```
