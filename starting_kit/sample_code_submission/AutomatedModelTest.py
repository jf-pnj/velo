import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import datetime 

from libscores import get_metric

class ModelTesting:
    '''
    This class automates the process of testing multiple machine learning models on
    a dataset.
    '''
    def __init__(self, 
                 X_train, 
                 y_train, 
                 models_list,
                 models_name,
                 preprocessing_name = None, 
				 scoring_function = None):
        '''
        This constructor initialises the datasets, scoring function, and models
        such that they can be used by the other methods.
		
		The default scoring function (scoring_function = None) is the provided one.
        '''
        self.X_train = X_train
        self.y_train = y_train
        if scoring_function == None:
            _, self.scoring_function = get_metric()
        else:
            self.scoring_function = scoring_function
        self.models_list = models_list
        self.models_name = models_name
        self.preprocessing_name = preprocessing_name
                
    def test_crossval(self):
        
        '''
        This function tests the different models that the user wants to test using
        the `cross_val_score` function in scikit-learn.
        
        It returns a pandas dataframe with the model names as index and the average
        cross-validation scores as values.
        '''
        results_crossval = pd.DataFrame(columns=['Average score'])
			
        for i in range(len(self.models_list)):
            print('Testing {}'.format(self.models_name[i]))
            scores = cross_val_score(self.models_list[i], 
                                     self.X_train, 
                                     self.y_train, 
                                     cv=10, 
                                     scoring=make_scorer(self.scoring_function))
            score = np.mean(scores)
            results_crossval.loc[self.models_name[i]] = score
            
        return results_crossval
    
    def best_model(self):
        '''
        This function first runs the `self.test_crossval` function to obtain the
        results dataframe, and afterwards selects the best model from the models list,
        which is what it returns.
        '''
        results = self.test_crossval()
        the_date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        if self.preprocessing_name:
            results.to_csv('result_files/ModelTesting Results_{}_{}.csv'.format(self.preprocessing_name,
                                                                               the_date))
        else:
            results.to_csv('result_file/ModelTesting Results.csv')
        best_model_name = results['Average score'].idxmax()
        best_model_idx = self.models_name.index(best_model_name)
        best_model = self.models_list[best_model_idx]
        return best_model, best_model_name, results