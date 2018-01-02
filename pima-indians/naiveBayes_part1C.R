wdat<-read.csv('/home/gitika.jain/Desktop/AML/assignment1/pima-indians-diabetes.data.csv', header=FALSE)
library(klaR)
library(caret)


bigx<-wdat[,-c(9)]
bigy<-as.factor(wdat[,9])
#wtd<-createDataPartition(y=bigy, p=.8, list=FALSE)
indices1 <- read.csv('/home/gitika.jain/Desktop/AML/assignment1/tr_indices.csv', header = FALSE)
wtd <- as.matrix(indices1)
trax<-bigx[wtd,]
tray<-bigy[wtd]
model<-train(trax, tray, 'nb', trControl=trainControl(method='cv', number=10))
teclasses<-predict(model,newdata=bigx[-wtd,])
confusionMatrix(data=teclasses, bigy[-wtd])


