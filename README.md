# TibFold

TibFold is a small Python library to calculate test error while performing hyper-parameter optimization when using k-fold cross validation.


## Installation

Tibfold can be installed by means of pip the Python standard package manager:

```
pip install tibfold
```

## Wait, what?

In machine learning, we train models to fit data. Usually, when we say data we are refering to a set of examples to learn from. Such models incurr in some error considering the data used for training, this is what we call training error. But, we are actually more interested in finding out how much error our model incurr when dealing with data not used during training. This second error is what we call test error.

### Training and test split

When we have a limited amount of data, what we usually do is to split the data in two disjunt sets, the so-called training set and test set. Sometimes, we randomly select $60\%$ of the data for training and $40\%$ for calculating test error.

### The k-fold cross validation algorithm

Splitting the data in two, as in the training/test split approach, will reduce the number of examples available for training. We want to have as much data as possible for training and as much data as possible for testing as well. What can we do when we have a small dataset?

The k-fold cross validation algorithm randomly splits the data in `k` equally sized subsets of examples, also called folds. Then, it uses one of the folds for testing, while the union of the other folds are used for training. The previous procedure is done a number of `k` times and each time a different fold is selected for testing. The average of the test error obtained the `k` times is the test error. 

Notice that using this approach, we ensure using all data for training and testing as well. However, the computational complexity of doing so is much higher than the previous training/test split.

### Training, validation and test split

Let's say that we have several machine learning models. How we select the best one among all? Do we use test error for this?

First, is worth to say that model **model selection** is a type of **hyper-parameter optimization**. Also, we are doing hyper-parameter optimization when we are not sure of which algorithm will be better, for example, when training a neural network (SGD, ADAM, AdaGrad, RMsProp, etc.) or how many levels do a decision tree should have.

If we use test error for model selection, or more generally for hyper-parameter optimization, we are selecting the hyper-parameters that best performs in that specific test set, but, what about in another set not seen during training or during model selection?

Answer: We don't know.

To overcome this, if we have enough data, we can split it in three subsets: training set, validation set and test set. We train all models using the training set. Then, we select the best performing model using the validation set and finally, the test error of the best performing model is calculated using the test set.

What can we do if we have a small dataset and splitting our data in three is not practical?

Answer: nested k-fold cross validation or the **Tibshirani-Tibshirani** method.

### Nested k-fold cross validation

The nested k-fold cross validation is similar to the previously introduced k-fold cross validation approach. Here, we are considering several machine learning models. For each model, we split the data in `k` folds. Then, we keep isolated the test fold and the remaining data (which is used for training) is again splitted in `k` folds. This inner k-fold cross validation is used for calculating validation error for each model and is repeated `k` times, changing the fold we keep isolated in each iteration. With the fold we let separated, we calculate the test error of each model.

Here is important that we select the best perfoming model only considering the validation error, not looking and the test error. Doing otherwise, we may endup with a biased estimate of the test error. The major drawback of this approach is of course the very high computational complexity of performing a k-fold cross validation procedure inside another.

### The Tibshirani-Tibshirani method

The Tibshirani-Tibshirani method is an aproach for calculating test error while performing hyper-parameter optimization using k-fold cross validation. It was introduced in 2009 by Ryan J. Tibshirani and Robert Tibshirani [1]. The difference here is that the computational complexity is the same that a regular k-fold cross validation.

The general idea is to perform a regular k-fold cross validation for each set of hyper-parameters. Then we select the best performing set of hyper-parameters based in the test error as usual, we consider this test error as validation error. The actual test error, will be the already calculated validation error plus some bias term. The bias is calculated as:


b = 1 / k \sum_{i=1}^k{e_i(\hat{\Omega}) - e_i(\hat{\Omega}_i)}


where $\hat{\Omega}$ is the best performing set of hyper-parameters and $\hat{\Omega}_i$ is the best performing set of hyper-parameters but only when considering the i-th fold. We also denote as $e_i(.)$ to the test error obtained when the i-th fold was used as test set. Note that all the required information can be gathered by simply performing a regular k-fold cross validation.

Actually, practical results have shown that the bias estimate provided by the Tibshirani-Tibshirani method is quite similar to the one obtained by the more expensive nested k-fold cross validation [2].

## Usage

TibFold comes as a single small class that implements k-fold cross validation and calculates the statistics related to the Tibshirani-Tibshirani method. To import TibFold in your project it can be simply done as:


```python
import TibFold as tf
```

Further, we need to create a TibFold class instance. When creating a new TibFold class, we need to provide:

- The training set input features in the form of a `numpy` bi-dimensional array where the columns represents input features and the rows represents the examples. 

- The training set output features in the form of a `numpy` bi-dimensional array where the columns represents output features and the rows represents the examples.

- The number of splits of the k-fold cross validation method.

- A score metric to calculate the error between the output features and model output. By default, TibFold uses the Mean Squared Error to this end. Actually, this should be a function with the following signature: `scorer(y1, y2)` where `y1` and `y2` are two `numpy` bi-dimensional array where the columns represents output features and the rows represents the examples of the ground truth and the model output respectively.

The following example shows how to instanciate a TibFold class using a randomly generated dataset of fifty examples, four input features and one output variable:


```python
import numpy as np

# Input examples
X = np.random.randint(0, 50, size=(50, 4))

# Output examples
y = np.random.randint(0, 50, size=(50, 1))

# Number of splits
k = 5

# TibFold class
tib = tf.TibFold(X, y, k)
```

For hyper-parameter optimization, we need to call the `cross_val_score(estimator)` method. The `estimator` parameter should be a class with at least an `estimator.fit(X, y)` method and a `estimator.predict(X)` method.

- The estimator class represents a machine learning model with its own set of hyper-parameters.
- The `estimator.fit(X, y)` method should train the model with the given dataset of X and y.
- The `estimator.predict(X)` method should return the model outputs for the given dataset X.

This enables TibFold `cross_val_score(estimator)` to be feeded with many models provided by popular libraries such as SkitLearn. The following examples how to perform hyper-parameter optimization of a decisión tree using SkitLearn.


```python
from sklearn.tree import DecisionTreeRegressor

models = [DecisionTreeRegressor(max_depth=depth) for depth in range(10, 50, 10)]

for model in models:
    tib.cross_val_score(model)
```

Finally, we can obtain training error, validation error and test error of the best performing model:


```python
print("Training error: " + str(tib.get_train_error()))
print("Validation error: " + str(tib.get_test_error()))
print("Test error: " + str(tib.get_test_error() + tib.get_bias()))
```

In case we may want to obtain the best performing model and its set of hyper-parameters, we can use the `get_best()` method, that will return the best model that have been provided via `cross_val_score(estimator)` method.


```python
tib.get_best()
```


## References

1. R. J. Tibshirani and R. Tibshirani. A bias correction for the minimum error rate in cross-validation, The Annals of Applied Statistics, págs. 822-829, 2009, ISSN: 19326157. URL: http://www.jstor.org/stable/30244266.

2. Ioannis Tsamardinos, Amin Rakhshani and Vincenzo Lagan. Performance-Estimation Properties of Cross-Validation-Based Protocols with Simultaneous Hyper-Parameter Optimization.
