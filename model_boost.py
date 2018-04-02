import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

params = {'n_estimators': 1000, 'max_depth': 10, 'random_state': 2,
                   'min_samples_split': 4, 'verbose': 1, 'max_leaf_nodes':3}

X_train = np.load('../X_train.npy').astype(np.float32)
Y = np.load('../Y.npy').astype(np.float32)
X_test = np.load('../X_test.npy').astype(np.float32)
X_train = X_train[:-219, :]

inds = np.any(np.isnan(X_train), axis=1)
X_train = X_train[~inds]
Y = Y[~inds]

inds = np.arange(Y.shape[0])
np.random.shuffle(inds)
X_train = X_train[inds]
Y = Y[inds]

# X_train = np.random.normal(0, 1, [100, 10])
# X_test = np.random.normal(0, 1, [100, 10])
# Y = np.random.randint(0, 2, [100])

clf = ensemble.GradientBoostingClassifier(**params)
clf._loss = log_loss
kf = KFold(n_splits=4)
scores = []
predict = []
for train, test in kf.split(X_train):
    clf.fit(X_train[train], Y[train])
    pred = clf.predict_proba(X_train[test])[:, 1]
    score = log_loss(Y[test], pred)
    scores.append(score)
    print('score', score)
    predict.append(clf.predict_proba(X_test)[:, 1])

scores = np.array(scores)
print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))
predict = np.array(predict)
predict = predict.mean(axis=0)
np.save('../predict.npy', predict)

submission = pd.read_csv('../data/sample_submission.csv')
submission.is_fake = predict
submission.to_csv('../submission1.csv', index=False)

