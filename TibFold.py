import math
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

class TibFold:

  def __init__(self, X, y, n_splits, scorer = mean_squared_error):
    self._X = X
    self._y = y
    self._best_test_error = math.inf
    self._best_train_error = math.inf
    self._n_splits = n_splits
    self._fbest_values = [math.inf] * n_splits
    self._kf = KFold(n_splits = n_splits)
    self._scorer = scorer

  def cross_val_score(self, estimator):
    test_sum = 0
    train_sum = 0
    fold = 0

    for train_index, test_index in self._kf.split(self._X):
      X_train, X_test = self._X[train_index], self._X[test_index]
      y_train, y_test = self._y[train_index], self._y[test_index]

      estimator.fit(X_train, y_train)

      y_train_pred = estimator.predict(X_train)
      y_test_pred = estimator.predict(X_test)
      
      train_error = self._scorer(y_train, y_train_pred)
      test_error = self._scorer(y_test, y_test_pred)

      train_sum += train_error
      test_sum += test_error

      if test_error < self._fbest_values[fold]:
        self._fbest_values[fold] = test_error

      fold += 1
    
    train_error = train_sum / self._n_splits
    test_error = test_sum / self._n_splits

    if test_error < self._best_test_error:
      self._best_test_error = test_error
      self._best_train_error = train_error
      self._estimator = estimator

    return test_error

  def get_bias(self):
    fold_sum = 0

    for i in range(0, self._n_splits):
      fold_sum += self._best_test_error - self._fbest_values[i]

    return fold_sum / self._n_splits

  def get_test_error(self):
    return self._best_test_error

  def get_train_error(self):
    return self._best_train_error

  def get_best(self):
    return self._estimator
