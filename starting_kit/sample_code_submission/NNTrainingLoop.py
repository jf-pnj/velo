import torch as th
import torch.nn.functional as F
import torch.nn as nn

from NNRegressor import NeuralNetwork

class trainingLoopRegression():
    def __init__(self):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        
    def train(self, train_data, train_labels, dev_data, dev_labels, optimizer, loss_func, dev_scorer, n_epochs, batch_size, gradient_clip, input_dim=None, layer_dims=None, dropout_value=None, model = None):        
        if model:
            self.model = model.to(self.device)
        else:
            self.model = NeuralNetwork(input_dim, layer_dims, dropout_value).to(self.device)
                
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
                
                cost += loss.item()
            
            mean_loss = cost / (len(train_data)/batch_size+1)
            dev_score = self.validation_score(self.model, dev_data, dev_labels, dev_scorer)
            
            print('mean loss: ', mean_loss)
            print('dev score: ', dev_score)
            
            loss_list.append(mean_loss)
            dev_score_list.append(dev_score)
        
        return self.model, loss_list, dev_score_list
    
    def validation_score(self, model, X_dev, y_dev, dev_scorer):
        outputs = model(X_dev.to(self.device))
        loss = dev_scorer(y_dev.detach().cpu().numpy(), outputs.detach().cpu().numpy())
        return loss / len(X_dev)
            
    def plot_graphs(self,
                    mean_losss,
                    dev_scores, 
                    n_hidden_layers):
        plt.plot([i for i in range(EPOCHS)],mean_losss, label='Mean loss')
        plt.plot([i for i in range(EPOCHS)],dev_accus, label='Performance on dev')
        plt.xlabel('Epochs')
        plt.title('{} hidden layers, {} embedding_size'.format(n_hidden_layers,
                                                              embedding_size))
        plt.legend()
        plt.savefig('{}HiddenLayer{}.png'.format(n_hidden_layers,
                                                embedding_size))
        plt.show()