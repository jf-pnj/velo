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
        self.PCA = PCA(n_components = 20)
        
    def fit(self, X, y):
        '''
        This function fits the Isolation Forest to the data
        PCA is fit and 

        '''
        self.IF.fit(X, y)
        return self

    def transform(self, X, y):
        try:
            res = self.IF.predict(X)
        except NotFittedError as e:
            print(repr(e))
            
        outlier_idx = []
        for i in range(0, len(res)):
            if res[i] == -1:
                outlier_idx.append(i)
        
        # outlier_prct = len(outlier_idx) / len(res) *100
        
        self.X_IF = np.delete(X, outlier_idx ,axis=0), 
        self.y_IF = np.delete(y, outlier_idx, axis=0)
        
        return self.X_IF[0], self.y_IF
        
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        X_process, y_process = self.transform(X, y)
        return X_process, y_process
        
def test():
    # Load votre model
    prepro = PreProcess()
    # 1 - créer un data X_random et y_random fictives: utiliser https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.random.rand.html
	# 1000 examples of 60 rows (same number of variables as the original dataset
    X_random = np.random.rand(1000, 60)
    y_random = np.random.rand(1000, 1)
    # 2 - Tester l'entrainement avec mod.fit(X_random, y_random)
    X_prepro, y_prepro = prepro.fit_transform(X_random, y_random)
    # 3 - Test la prediction: mod.predict(X_random)
    # Pour tester cette fonction *test*, il suffit de lancer la commande ```python sample_code_submission/model.py```
    #print(X_prepro)
    #print(y_prepro)
    #print(len(X_prepro))
    #print(len(y_prepro))

if __name__ == "__main__":
    test()        