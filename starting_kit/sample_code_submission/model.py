import numpy as np  # We recommend to use numpy arrays
from os.path import isfile
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor


class model(BaseEstimator):
    def __init__(self):
        '''
        This constructor is supposed to initialize data members.
        Use triple quotes for function documentation.

        The preprocess and model are made variable so this script can
        be incorporated in pipelines more easily.
        '''
        self.num_train_samples = 38563
        self.num_feat = 59
        self.num_labels = 1
        self.is_trained = False
        self.preprocess = PCA(n_components=0.999)  # Ex. PCA()
        self.mod = GradientBoostingRegressor(random_state=42, criterion = 'friedman_mse',
                             learning_rate = 0.15143814974272138,
                             loss = 'ls', max_depth = 10,
                             min_samples_split = 6, n_estimators = 369)
                            # Initalizing the model # Ex. DecisionTreeRegressor()

    def fit(self, X, y):
        '''
        This function should train the model parameters.
        Here we do nothing in this example...
        Args:
            X: Training data matrix of dim num_train_samples * num_feat.
            y: Training label matrix of dim num_train_samples * num_labels.
        Both inputs are numpy arrays.
        For classification, labels could be either numbers 0, 1, ... c-1 for c classe
        or one-hot encoded vector of zeros, with a 1 at the kth position for class k.
        The AutoML format support on-hot encoding, which also works for multi-labels problems.
        Use data_converter.convert_to_num() to convert to the category number format.
        For regression, labels are continuous values.
        '''
        if X.ndim > 1: self.num_feat = X.shape[1]
        if y.ndim > 1: self.num_labels = y.shape[1]

        X_preprocess = self.preprocess.fit_transform(X)
        self.mod.fit(X_preprocess, y)
        self.is_trained = True

    def predict(self, X):
        '''
        This function should provide predictions of labels on (test) data.
        Here we just return zeros...
        Make sure that the predicted values are in the correct format for the scoring
        metric. For example, binary classification problems often expect predictions
        in the form of a discriminant value (if the area under the ROC curve it the metric)
        rather that predictions of the class labels themselves. For multi-class or multi-labels
        problems, class probabilities are often expected if the metric is cross-entropy.
        Scikit-learn also has a function predict-proba, we do not require it.
        The function predict eventually can return probabilities.
        '''
        num_test_samples = X.shape[0]
        if X.ndim > 1: num_feat = X.shape[1]
        y = np.zeros([num_test_samples, self.num_labels])

        X_preprocess = self.preprocess.transform(X)
        y = self.mod.predict(X_preprocess)
        return y

    def save(self, path="./"):
        pass

    def load(self, path="./"):
        pass


def test():
    # Load votre model
    mod = model(PCA(), DecisionTreeRegressor())
    # 1 - créer un data X_random et y_random fictives: utiliser https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.rand.html
    # 1000 examples of 60 rows (same number of variables as the original dataset
    X_random = np.random.rand(1000, 60)
    y_random = np.random.rand(1000, 1)
    # 2 - Tester l'entrainement avec mod.fit(X_random, y_random)
    mod.fit(X_random, y_random)
    # 3 - Test la prediction: mod.predict(X_random)
    predictions = mod.predict(X_random)
    # Pour tester cette fonction *test*, il suffit de lancer la commande ```python sample_code_submission/model.py```


if __name__ == "__main__":
    test()