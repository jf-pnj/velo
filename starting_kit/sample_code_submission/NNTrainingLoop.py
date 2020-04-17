import matplotlib.pyplot as plt
import torch as th

from NNRegressor import NeuralNetwork

class trainingLoopRegression():
    '''
    This class is a training loop for the PyTorch Neural Network that is input 
    into. 
    '''
    def __init__(self):
        '''
        This constructor only sets the device (cpu or gpu) used for the PyTorch 
        model. All the other variables are input through the function arguments 
        of the methods of this class.
        '''
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
    def train(self, train_data, train_labels, dev_data, dev_labels, optimizer, loss_func, dev_scorer, n_epochs, batch_size, gradient_clip, input_dim=None, layer_dims=None, dropout_value=None, model = None):        
        '''
        This function trains a provided Neural Network model on the training 
        data using batches.


        Parameters
        ----------
        train_data : The training data in Tensor format
        train_labels : The training labels in Tensor format
        dev_data : The validation data in Tensor format
        dev_labels : The validation labels in Tensor format
        optimizer : The PyTorch optimizer used for training
        loss_func : The differentiable loss function used for training
        dev_scorer : The metric to evaluate the validation set with
        n_epochs : The number of epochs
        batch_size : The batch size
        gradient_clip : The bound for gradient clipping
        input_dim : optional, number of variables in the data
        layer_dims : optional list for number of neurons per layer of the
                     Neural Network
        dropout_value : optional, dropout value of the Neural Network
        model : optional, the PyTorch Neural Network to train. If None, the
                function constructs its own network (FUNCTIONALITY STILL TO BE
                EXPANDED)             

        Returns
        -------
        model : The trained model
        loss_list : List of the loss values on the training set during training
        dev_score_list : List of scores on the development set during training

        '''
        self.n_epochs = n_epochs
        if model:
            self.model = model.to(self.device)
        else:
            self.model = NeuralNetwork(input_dim, layer_dims, dropout_value).to(self.device)
                
        self.model.apply(self.init_weights)
        self.optimizer = optimizer
        loss_list = []
        dev_score_list = []
        
        for epoch in range(n_epochs):
            print('Starting epoch: {}'.format(epoch))
            self.model.train()
            cost = 0
            
            for first in range(0, len(train_data), batch_size):
                self.optimizer.zero_grad()                                     
                
                batch_input = th.cat(
                    [
                        example.reshape(1, 59)
                        for example in train_data[first:first + batch_size]                        
                    ],
                        dim=0
                ).to(self.device)
                
                batch_labels = train_labels[first:first + batch_size].unsqueeze(1).to(self.device)
				
                output = self.model(batch_input)
                                                                
                loss = loss_func(output, batch_labels)
                                
                loss.backward()
                
                th.nn.utils.clip_grad_value_(self.model.parameters(), gradient_clip)
                
                self.optimizer.step()
                
                cost += loss.detach().item() * batch_size / len(train_data[0])
            
            mean_loss = cost / (len(train_data)/batch_size+1)
            dev_score = self.validation_score(self.model, dev_data, dev_labels, dev_scorer)
            
            print('mean loss: ', mean_loss)
            print('dev score: ', dev_score)
            
            loss_list.append(mean_loss)
            dev_score_list.append(dev_score)
        
        return self.model, loss_list, dev_score_list
    
    def init_weights(self, m):
        '''
        This function initializes the weights and biases of all the layers in the
        network. This function automatically only initializes the nn.Linear type
        objects due to the if-statement check. It uses kaiming initialization and
        it initializes the biases with zeros.
        '''
        if type(m) == th.nn.Linear:
            th.nn.init.kaiming_uniform_(m.weight.data) 
            th.nn.init.zeros_(m.bias.data)   

    
    def validation_score(self, model, X_dev, y_dev, dev_scorer):
        '''
        This method calculates the score of the model on the development set
        during training.
        '''
        outputs = model(X_dev.to(self.device))
        loss = dev_scorer(y_dev.detach().cpu().numpy(), outputs.detach().cpu().numpy())
        return loss / len(X_dev)
            
    def plot_graphs(self,
                    mean_losss,
                    dev_scores, 
                    n_hidden_layers,
                    n_epochs):
        '''
        This method plots the training loss and performance on the development
        set achieved during the training process.
        '''
        plt.plot([i for i in range(n_epochs)],mean_losss, label='Mean loss')
        plt.plot([i for i in range(n_epochs)],
                 dev_scores, 
                 label='Performance on dev')
        plt.xlabel('Epochs')
        plt.title('{} hidden layers, {} embedding_size'.format(n_hidden_layers))
        plt.legend()
        plt.savefig('{}HiddenLayer{}.png'.format(n_hidden_layers))
        plt.show()