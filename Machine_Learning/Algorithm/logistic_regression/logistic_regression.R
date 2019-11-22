# CSE 6242: Data and Visual Analytics 
# HW3
# Name: Zuolin Liu
# GT account: zliu653

# remove all variables in global environment
rm(list=ls(all=TRUE))
library(ggplot2)

# 1. Data Preprocessing
# read data
train = as.data.frame(t(read.csv("mnist_train.csv", header=FALSE)))
test = as.data.frame(t(read.csv("mnist_test.csv", header=FALSE)))

# Partition the training set and test set for classification of 0, 1 and 3, 5 classes
train_0_1 = train[which(train$V785==0 | train$V785==1),]
test_0_1 = test[which(test$V785==0 | test$V785==1),]
train_3_5 = train[which(train$V785==3 | train$V785==5),]
test_3_5 = test[which(test$V785==3 | test$V785==5),]

# print the number of sample for each partition 
print(paste('Number of training data for 0 and 1: ', dim(train_0_1)[1]))
print(paste('Number of test data for 0 and 1: ', dim(test_0_1)[1]))
print(paste('Number of training data for 3 and 5: ', dim(train_3_5)[1]))
print(paste('Number of test data for 3 and 5: ', dim(test_3_5)[1]))

# Separate the true class label from all the partitions created
train_label_0_1 = train_0_1$V785
train_label_3_5 = train_3_5$V785
test_label_0_1 = test_0_1$V785
test_label_3_5 = test_3_5$V785
train_data_0_1 = train_0_1[, 1:784]
train_data_3_5 = train_3_5[, 1:784]
test_data_0_1 = test_0_1[, 1:784]
test_data_3_5 = test_3_5[, 1:784]

# Visualize 1 image from each class
rotate = function(x) t(apply(x, 2, rev)) 
plot_image = function(data, label){
  data = matrix(unlist(data), nrow = 28, byrow=FALSE)
  image(rotate(data), col=gray(0:255/255), main=label)
}
data_0 = train_data_0_1[train_label_0_1 == 0, ][1,]
data_1 = train_data_0_1[train_label_0_1 == 1, ][1,]
data_3 = train_data_3_5[train_label_3_5 == 3, ][1,]
data_5 = train_data_3_5[train_label_3_5 == 5, ][1,]
plot_image(data_0, '0')
plot_image(data_1, '1')
plot_image(data_3, '3')
plot_image(data_5, '5')

# 3. Implementation
# map labels
train_label_0_1[train_label_0_1 == 0] = -1
test_label_0_1[test_label_0_1 == 0] = -1
train_label_3_5[train_label_3_5 == 3] = -1
test_label_3_5[test_label_3_5 == 3] = -1
train_label_3_5[train_label_3_5 == 5] = 1
test_label_3_5[test_label_3_5 == 5] = 1

# convert traning data and test data into matrices
train_data_0_1 = data.matrix(train_data_0_1)
test_data_0_1 = data.matrix(test_data_0_1)
train_data_3_5 = data.matrix(train_data_3_5)
test_data_3_5 = data.matrix(test_data_3_5)

# train function
train = function(data, labels, alpha, num_iter=50, batch_size=256){
  data = add_constant(data)  # append a constant 1 to data
  num_data = nrow(data)
  num_feature = ncol(data)
  num_batch = ceiling(num_data/batch_size)
  theta = rnorm(num_feature)
  theta = matrix(theta, nrow = num_feature, byrow = TRUE)
  labels = matrix(labels, nrow = num_data, ncol = 1)
  # shuffle the trainging data
  shuffle_idx = sample.int(num_data)
  data = data[shuffle_idx,]
  labels = labels[shuffle_idx]
  # training
  for (i in seq(1, num_iter)){
    for (j in seq(1, num_batch)){
      start = (j-1)*batch_size + 1
      end = j*batch_size
      if (end > num_data){
        end = num_data
      }
      X = data[start:end, ]
      y = labels[start:end]
      exp = exp(y * (X %*% theta))
      Y = replicate(num_feature, y)
      EXP = replicate(num_feature, as.vector(exp))
      theta_delta = colSums(Y * X * EXP / (1 + EXP))
      theta_delta = matrix(theta_delta, nrow = num_feature, byrow = TRUE)
      theta = theta - theta_delta * alpha
    }
    loss = sum(log(1 + exp(labels * (data %*% theta))))
    pred = predict(data, theta)
    accuracy = accuracy(labels, pred)
    print(paste("Epoch: ", i, 'loss', loss, 'accuracy: ', accuracy))
  }
  return(theta)
}

# predict function
predict = function(data, theta){
  if (dim(data)[2] < dim(theta)[1]){
    data = add_constant(data)
  }
  y = matrix(0L, nrow = nrow(data), ncol = 1)
  p = 1/(1 + exp(data %*% theta))
  y[p >= 0.5] = 1
  y[p < 0.5] = -1
  return(y)
}

# add a constant to data
add_constant = function(data){
  data = cbind(1, data)
  return(data)
}

# 4. Modeling
accuracy = function(labels, labels_pred){
  accuracy = sum((labels- labels_pred) == 0)/length(labels_pred) 
  return(accuracy)
}

model = function(train_data, train_labels, test_data, test_labels, alpha){
  theta = train(train_data, train_labels, alpha, num_iter=1)
  train_data = add_constant(train_data)
  test_data = add_constant(test_data)
  train_pred = predict(train_data, theta)
  train_acc = accuracy(train_labels, train_pred)
  test_pred = predict(test_data, theta)
  test_acc = accuracy(test_labels, test_pred)
  result = list(theta = theta, train_acc = train_acc, test_acc = test_acc)
  return(result)
}

# Prepare result for Q3
# helper function to visulize 2 correct predictions and 2 incorrect prediction
plot_prediction=function(data, label, theta){
  pred = predict(data, theta)
  incorrect_idx = which(pred != label)
  correct_idx = which(pred == label)
  incorrect_data_1 = data[incorrect_idx[1],]
  incorrect_data_2 = data[incorrect_idx[length(incorrect_idx)],]
  correct_data_1 = data[correct_idx[1],]
  correct_data_2 = data[correct_idx[length(correct_idx)],]
  plot_image(incorrect_data_1, 
             paste('Incorrect prediction, True label:', label[incorrect_idx[1]],
                   'Predicted label:', pred[incorrect_idx[1]]))
  plot_image(incorrect_data_2, 
             paste('Incorrect prediction, True label:', label[incorrect_idx[length(incorrect_idx)]],
                   'Predicted label:', pred[incorrect_idx[length(incorrect_idx)]]))
  plot_image(correct_data_1, 
             paste('Correct prediction, True label:', label[correct_idx[1]],
                   'Predicted label:', pred[correct_idx[1]]))
  plot_image(correct_data_2, 
             paste('Correct prediction, True label:', label[correct_idx[length(correct_idx)]],
                   'Predicted label:', pred[correct_idx[length(correct_idx)]]))
}

# Visualization of 2 correct and 2 incorrect predictions each for 0/1 training sets
theta_train_0_1 = train(train_data_0_1, train_label_0_1, 0.01)
plot_prediction(train_data_0_1, train_label_0_1, theta_train_0_1)

# Visualization of 2 correct and 2 incorrect predictions each for 3/5 training sets
theta_train_3_5 = train(train_data_3_5, train_label_3_5, 0.03)
plot_prediction(train_data_3_5, train_label_3_5, theta_train_3_5)

# prepare result for Q4
# helper function to calculate accuracies for different learning rates
accu_lr=function(train_data, train_label, test_data, test_label, alphas){
  acc_train_mean = vector()
  acc_test_mean = vector()
  acc_train_std = vector()
  acc_test_std = vector()
  for (i in seq(length(alphas))){
    new_acc_train = vector()
    new_acc_test = vector()
    for (j in seq(5)){
      print(paste('Learning rate: ', alphas[i], 'Run #', j))
      result = model(train_data, train_label, test_data, test_label, alphas[i])
      new_acc_train = c(new_acc_train, result$train_acc)
      new_acc_test = c(new_acc_test, result$test_acc)
    }
    acc_train_mean = c(acc_train_mean, mean(new_acc_train))
    acc_train_std = c(acc_train_std, sd(new_acc_train))
    acc_test_mean = c(acc_test_mean, mean(new_acc_test))
    acc_test_std = c(acc_test_std, sd(new_acc_test))
  }
  result = data.frame(alphas, acc_train_mean, acc_train_std, acc_test_mean, acc_test_std)
  return(result)
}


# helper function to plot accuracy as errorbar
plot_accuracy_lr=function(data, title){
  ggplot(data=data,aes(x = alphas,y=acc_train_mean))+
    geom_line(data = data,aes(x=alphas,y=acc_train_mean,colour="train"))+
    geom_errorbar(data = data,aes(ymin=acc_train_mean-acc_train_std,ymax=acc_train_mean+acc_train_std),colour="red")+
    geom_line(data = data,aes(x=alphas,y=acc_test_mean,colour="test"))+
    geom_errorbar(data = data,aes(ymin=acc_test_mean-acc_test_std,ymax=acc_test_mean+acc_test_std),colour="blue")+
    scale_x_log10() + labs(x='Learning rate') + labs(title = title) + labs(y = 'Accuracy') +
    scale_colour_manual(values=c(train='red',test='blue'))
}
alpha_0_1 = c(0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.1)
acc_lr_0_1 = accu_lr(train_data_0_1, train_label_0_1, test_data_0_1, test_label_0_1, alpha_0_1)
plot_accuracy_lr(acc_lr_0_1, 'Accuracy vs learning rate for 0 and 1')

alpha_3_5 = c(0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064)
acc_lr_3_5 = accu_lr(train_data_3_5, train_label_3_5, test_data_3_5, test_label_3_5, alpha_3_5)
plot_accuracy_lr(acc_lr_3_5, 'Accuracy vs learning rate for 3 and 5')

# 5.Learning curve
data_fraction = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1)
# helper function to calculate accuracies for different training data size
accu_size=function(train_data, train_label, test_data, test_label, data_fraction){
  num_data = dim(train_data)[1]
  acc_train_mean = vector()
  acc_test_mean = vector()
  acc_train_std = vector()
  acc_test_std = vector()
  for (i in seq(length(data_fraction))){
    shuffle_idx = sample.int(num_data * data_fraction[i])
    data = train_data[shuffle_idx,]
    label = train_label[shuffle_idx]
    new_acc_train = vector()
    new_acc_test = vector()
    for (j in seq(5)){
      print(paste('Sample fraction: ', data_fraction[i], 'Run #', j))
      result = model(data, label, test_data, test_label, 0.01)
      new_acc_train = c(new_acc_train, result$train_acc)
      new_acc_test = c(new_acc_test, result$test_acc)
    }
    acc_train_mean = c(acc_train_mean, mean(new_acc_train))
    acc_train_std = c(acc_train_std, sd(new_acc_train))
    acc_test_mean = c(acc_test_mean, mean(new_acc_test))
    acc_test_std = c(acc_test_std, sd(new_acc_test))
  }
  result = data.frame(data_fraction, acc_train_mean, acc_train_std, acc_test_mean, acc_test_std)
  return(result)
}

plot_accuracy_size=function(data, title){
  ggplot(data=data,aes(x = data_fraction, y=acc_train_mean))+
    geom_line(data = data,aes(x=data_fraction, y=acc_train_mean,colour="train")) +
    geom_errorbar(data = data,aes(ymin=acc_train_mean-acc_train_std,ymax=acc_train_mean+acc_train_std),colour="red")+
    geom_line(data = data,aes(x=data_fraction,y=acc_test_mean,colour="test")) +
    geom_errorbar(data = data,aes(ymin=acc_test_mean-acc_test_std,ymax=acc_test_mean+acc_test_std),colour="blue")+
    labs(x = 'Sample fraction') + labs(y = 'Accuracy') + labs(title = title) +
    scale_colour_manual(values=c(train='red',test='blue'))
}

acc_size_0_1 = accu_size(train_data_0_1, train_label_0_1, 
                         test_data_0_1, test_label_0_1, data_fraction)
plot_accuracy_size(acc_size_0_1, 'Sample fraction vs learning rate for 0 and 1')

acc_size_3_5 = accu_size(train_data_3_5, train_label_3_5, 
                         test_data_3_5, test_label_3_5, data_fraction)
plot_accuracy_size(acc_size_3_5, 'Sample fraction vs learning rate for 3 and 5')




