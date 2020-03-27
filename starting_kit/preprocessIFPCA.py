import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

class PreProcess:
    def __init__(self):
        '''
        This constructor initializes both the Isolation Forest and the PCA
        '''
        self.IF = IsolationForest(random_state = 0)
        self.PCA = PCA(n_components = 5)
        
    def fit_IF(self, X, y):
        '''
        This function fits the Isolation Forest to the data
        
        '''
        self.IF.fit(X, y)

    def transform_IF(self, X, y=None):
        '''
        This function uses the fitted Isolation Forest to determine the 
        outliers in the data. The indexes of these outliers are determined.
        The datapoints with an Isolation Forest score of -1 are deleted from
        the datapoints. 
        '''  
        # Check if the model has already been fitted, and if not, throw an 
        # error
        try:
            res = self.IF.predict(X)
        except NotFittedError as e:
            print(repr(e))
            
        outlier_idx = []
        for i in range(0, len(res)):
            if res[i] == -1:
                outlier_idx.append(i)
        
        # outlier_prct = len(outlier_idx) / len(res) *100
        
        self.X_IF = np.delete(X, outlier_idx ,axis=0)
        if y is not None:
            self.y_IF = np.delete(y, outlier_idx, axis=0)
            return self.X_IF, self.y_IF
        else:
            return self.X_IF
        
    def fit_transform_IF(self, X, y):
        '''
        This function combines the fit_IF and transform_IF functions to fit
        and transform the input data using an Isolation Forest.
        '''
        self.fit_IF(X, y)
        X_process_IF, y_process_IF = self.transform_IF(X, y)
        return X_process_IF, y_process_IF
    
    def fit_PCA(self, X):
        '''
        This function fits the PCA model to the data X.
        '''
        self.PCA.fit(X)
        
    def transform_PCA(self, X):
        '''
        This function uses the fitted PCA model to transform the data X into
        an array with a reduced number of variables.
        '''
        # Check if the model has already been fitted, and if not, throw an 
        # error
        try:
            X_pca = self.PCA.transform(X)
        except NotFittedError as e:
            print(repr(e))
        return X_pca
    
    def fit_transform_PCA(self, X):
        '''
        This function combines the fit_PCA and transform_PCA functions to
        fit and transform the input data using PCA.
        '''
        self.fit_PCA(X)
        X_pca = self.transform_PCA(X)
        return X_pca
    
    def transform_IF_PCA(self, X, y):
        '''
        This function combines the transform_IF and transform_PCA functions in
        that order.
        '''
        X_IF, y_IF = self.transform_IF(X, y)
        X_PCA = self.transform_PCA(X_IF)
        return X_PCA, y_IF
    
    def fit_transform_IF_PCA(self, X, y):
        '''
        This function combines the fit_transform_IF and fit_transform_PCA
        functions in that order.
        '''
        X_prepro_IF, y_prepro = self.fit_transform_IF(X, y)
        X_prepro = self.fit_transform_PCA(X_prepro_IF)
        return X_prepro, y_prepro
        
def test():
    # Load votre model
    prepro = PreProcess()
    # 1 - créer un data X_random et y_random fictives: utiliser https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.rand.html
	# 1000 examples of 60 rows (same number of variables as the original dataset
    X_random = np.random.rand(1000, 60)
    y_random = np.random.rand(1000, 1)
    # 2 - Tester l'entrainement avec mod.fit(X_random, y_random)
    X_prepro, y_prepro = prepro.fit_transform_IF_PCA(X_random, y_random)
    # 3 - Test la prediction: mod.predict(X_random)
    # Pour tester cette fonction *test*, il suffit de lancer la commande ```python sample_code_submission/model.py```
    #print(X_prepro)
    #print(y_prepro)
    print(len(X_prepro))
    print(len(y_prepro))
    if len(X_prepro) == len(y_prepro):
        print("The test was a success!")

if __name__ == "__main__":
    test()        