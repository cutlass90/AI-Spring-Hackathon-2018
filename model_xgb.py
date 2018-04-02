import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold


def lgb_binary_loss(preds, dtrain):
    labels = dtrain.get_label()

    return "binary_loss", log_loss(labels, preds), False

# X_train = np.load('../X_train.npy').astype(np.float32)
# Y = np.load('../Y.npy').astype(np.float32)
# X_test = np.load('../X_test.npy').astype(np.float32)
# X_train = X_train[:-219, :]
#
# inds = np.any(np.isnan(X_train), axis=1)
# X_train = X_train[~inds]
# Y = Y[~inds]
#
# inds = np.arange(Y.shape[0])
# np.random.shuffle(inds)
# X_train = X_train[inds]
# Y = Y[inds]

X_train = pd.read_pickle('../X_train.pkl').as_matrix()
X_test = pd.read_pickle('../X_test.pkl').as_matrix()
Y = pd.read_pickle('../Y.pkl').as_matrix()
X_train = X_train[:-219]


# X_train = np.random.normal(0, 1, [1000, 10])
# X_test = np.random.normal(0, 1, [1000, 10])
# Y = np.random.randint(0, 2, [1000])

lgb_data_train = lgb.Dataset(
    X_train,
    label=Y,
    free_raw_data=False
)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.04,
    "scale_pos_weight": 5,
    "depth": 3,
    "num_leaves": 6
}


# kf = KFold(n_splits=4)
# scores = []
# predict = []
# for train, test in kf.split(X_train):
#     lgb_data_train = lgb.Dataset(
#         X_train[train],
#         label=Y[train],
#         free_raw_data=False
#     )
#     lgb_data_test = lgb.Dataset(
#         X_train[test],
#         label=Y[test],
#         free_raw_data=False
#     )
#     gbm = lgb.train(
#         params,
#         lgb_data_train,
#         num_boost_round=3000,
#         valid_sets=[lgb_data_test],
#         feval=lgb_binary_loss,
#         verbose_eval=10
#     )
#     pred = gbm.predict(X_train[test])
#     score = log_loss(Y[test], pred)
#     scores.append(score)
#     print('score', score)
#     predict.append(gbm.predict(X_test))
#
# scores = np.array(scores)
# print("score: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
# predict = np.array(predict)
# predict = predict.mean(axis=0)
# np.save('../predict.npy', predict)

gbm = lgb.train(
    params,
    lgb_data_train,
    num_boost_round=30000,
    valid_sets=[lgb_data_train],
    feval=lgb_binary_loss,
    verbose_eval=10
)

predict = gbm.predict(X_test)
submission = pd.read_csv('../data/sample_submission.csv')
submission.is_fake = predict
submission.to_csv('../submission4.csv', index=False)


for column, importance in zip(range(X_train.shape[1]), gbm.feature_importance()):
    print(column, importance)