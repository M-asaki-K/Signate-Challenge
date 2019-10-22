library("keras")

#-----------read csv file as compounds（学習、テスト用データ）--------------
#path <- file.choose() #自分でファイルを指定する場合
#path
compounds <- read.csv("C:\\Users\\Data science\\train.csv")
#compounds <- read.csv(path)

#-----------remove some columns if needed--------------
trimed.compounds <- compounds[,-c(1)] #サンプル番号を除去

#-----------select rows without empty cells---------
is.completes <- complete.cases(trimed.compounds) #空白セルを含む行を除去
is.completes

complete.compounds <- trimed.compounds[is.completes,]

#-----------read csv file as compounds（課題データ）--------------
#path.t <- file.choose() #自分でファイルを指定する場合
compounds.t <- read.csv("C:\\Users\\test.csv")
#compounds.t <- read.csv(path.t)

#-----------remove some columns if needed--------------
trimed.compounds.t <- compounds.t[,-c(1)]　#サンプル番号を除去

#-----------select rows without empty cells---------
is.completes.t <- complete.cases(trimed.compounds.t)
is.completes.t

complete.compounds.t <- trimed.compounds.t[is.completes.t,]

#-----------select x from the dataset-----------------
x.0 <- complete.compounds[,-c(1)] #学習用データの1列目は目的変数なので除去
x.0.t <- complete.compounds.t[,] #課題データはすべて説明変数
x.sds <- apply(x.0, 2, sd) #標準偏差0の列を除去
x.sds.t <- apply(x.0.t, 2, sd)

sd.is.not.0 <- x.sds != 0 
sd.is.0 <- x.sds == 0
sd.is.not.0.t <- x.sds.t != 0
sd.is.0.t <- x.sds.t == 0

x.0.eda <- x.0[, c(sd.is.0.t)]
x.0.eda.t <- x.0.t[, c(sd.is.0)]

#以降65行目までは、課題データor学習データでしか変化していない行を除去（必須の処理ではない）
row_sub = apply(x.0.eda[,-c(16)], 1, function(row) all(row ==0 )) 
row_sub_2 = x.0.eda[,c(16)] ==1
row_sub.t = apply(x.0.eda.t, 1, function(row) all(row ==0 ))
row_sub.t[row_sub.t == FALSE]

sd.is.not.0.f <- sd.is.not.0.t&sd.is.not.0
row_sub.f <- row_sub&row_sub_2
row_sub.t.f <- row_sub.t
x.0 <- x.0[row_sub.f, (sd.is.not.0.f)]
x.0.t <- x.0.t[, (sd.is.not.0.f)]

x.sds <- apply(x.0, 2, sd)
x.sds.t <- apply(x.0.t, 2, sd)

sd.is.not.0 <- x.sds != 0 
sd.is.not.0.t <- x.sds.t != 0
sd.is.not.0.f <- sd.is.not.0.t&sd.is.not.0
x.0 <- x.0[, (sd.is.not.0.f)]
x.0.t <- x.0.t[, (sd.is.not.0.f)]

nrow(x.0)
ncol(x.0)
nrow(x.0.t)
ncol(x.0.t)

x.1 <- rbind(x.0,x.0.t)
x.0.s <- as.data.frame(x.1[c(1:nrow(x.0)), ])
x.0.t.s <- as.data.frame(x.1[-c(1:nrow(x.0)), ])

#--------------------divide into test and training data----------------------
train_size = 1 #はじめは過学習が怖かったので、テストデータを別途区分けしていた

n = nrow(x.0.s)
#------------collect the data with n*train_size from the dataset------------
perm = sample(n, size = round(n * train_size))

train_data <- cbind.data.frame(x.0.s[perm, ])
test_data <- cbind.data.frame(x.0.s[-perm, ])
nrow(test_data)
#-----------select y from the dataset------------------
train_labels <- complete.compounds[row_sub.f, c(1)]
train_labels <- train_labels[perm]
test_labels <- complete.compounds[row_sub.f, c(1)]
test_labels <- test_labels[-perm]
test_labels

# Test data is *not* used when calculating the mean and std.

# Normalize training data
train_data <- scale(train_data) 

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(train_data, "scaled:center") 
col_stddevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

train_data[1, ] # First training sample, normalized

#-----------select x from the dataset-----------------
test_data.t <- cbind.data.frame(x.0.t.s[,])

# Test data is *not* used when calculating the mean and std.
# Use means and standard deviations from training set to normalize test set
test_data.t <- scale(test_data.t, center = col_means_train, scale = col_stddevs_train)

#2D CNNに入力するための次元変換（よく考えると1D CNNでやればよかった…）
dim(train_data) = c(nrow(train_data), 29, 127, 1)
dim(test_data.t) = c(nrow(test_data.t), 29, 127, 1)

input_shape = c(29, 127, 1)
batch_size = 128
epochs = 50

#学習と評価
model %>% fit(
  train_data, train_labels,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_split = 0.2,
)

#ベイズ最適化によるパラメータ調整
library(rBayesianOptimization)
Gauss_holdout <- function(okayama, tokyo, nagoya, aichi){
build_model <- function() {

  okayama = 0.5
  tokyo = 0.7
  nagoya = 0.1
  aichi = 0.1
  
  model = keras_model_sequential() %>%
    layer_conv_2d(filters = floor(okayama*40) + 2, kernel_size = c(3,3), activation = 'relu',
                  input_shape = c(29,127,1)) %>%
    layer_conv_2d(filters = floor(tokyo*40) + 2, kernel_size = c(3,3), activation = 'relu') %>%
    layer_max_pooling_2d(pool_size = c(2, 2)) %>%
    layer_dropout(rate = nagoya*0.9) %>%
    layer_flatten() %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dropout(rate = aichi*0.9) %>%
    layer_dense(units = 1)
  
  #モデルのコンパイル
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adadelta(),
    metrics = list("mean_absolute_error")
  )
  
  model
}

epochs <- 500
batch_size <- 128

model <- build_model()
model %>% summary()

# Display training progress by printing a single dot for each completed epoch.
print_dot_callback <- callback_lambda(
  on_epoch_end = function(epoch, logs) {
    if (epoch %% 80 == 0) cat("\n")
    cat(".")
  }
)    

lr_schedule<-function(epoch,lr) {
  lr=initial_lr/decay^((epoch-1)%%period)+1e-5
  lr
}
cb_lr<-callback_learning_rate_scheduler(lr_schedule)
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 50)

# Fit the model and store training stats
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 1,
  batch_size = batch_size,
  callbacks = list(early_stop)
)

Predi <- min(history$metrics$val_mean_absolute_error)*(-1)
list(Score=Predi, Pred=Predi)}

opt_svm <- BayesianOptimization(Gauss_holdout,bounds=list(okayama=c(0,1),tokyo=c(0,1),nagoya =c(0,1), aichi = c(0,1)),init_points=20, n_iter=30, acq='ucb', kappa = 2.576 / 2, verbose=TRUE)

plot(history)
gan[c(i), ] <- min(history$metrics$val_mean_absolute_error)
View(gan)
#}

train_predictions <- model %>% predict(train_data)
plot(train_predictions, train_labels)
r2train <- cor(train_predictions, train_labels)**2
r2train

test_predictions <- model %>% predict(test_data)
plot(test_predictions, test_labels)
r2test <- cor(test_predictions, test_labels)**2
r2test

test_predictions.t <- model %>% predict(test_data.t)
test_predictions.t <- as.matrix(test_predictions.t)
write.csv(test_predictions.t[1:13732], "C:/Users/withoutpcadlbest3.csv")
