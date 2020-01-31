import math
from sklearn.model_selection import KFold
from sklearn.metrics.scorer import check_scoring

class TibFold:
  _X
  _y  
  _n_splits
  _kf
  _best_error = math.inf
  _estimator

  def __init__(self, X, y, n_splits, scoring=None):
    self._X = X
    self._y = y
    self._n_splits = n_splits
    self._fbest_values = [math.inf] * n_splits
    self._kf = KFold(n_splits = n_splits)
    self._scorer = check_scoring(estimator, scoring=scoring)

  def cross_val_score(estimator):
    test_sum = 0
    fold = 0

    for train_index, test_index in _kf.split(_X):
      X_train, X_test = _X[train_index], _X[test_index]
      y_train, y_test = _y[train_index], _y[test_index]

      estimator.fit(X_train, y_train)
      
      train_error = self._scorer(estimator, X_train, y_train)
      test_error = self._scorer(estimator, X_test, y_test)
      test_sum += test_error

      if test_error < _fbest_values[fold] :
        _fbest_values[i] = test_error

      fold++
    
    test_error = test_sum / _n_splits

    if test_error < _best_error:
      _best_error = test_error
      _estimator = estimator

  def get_bias():
    fold_sum = 0

    for i in range(0, _n_splits):
      fold_sum += _best_error - _fbest_values[i]

    return fold_sum / _n_splits

  def get_best():
    return _estimator
