import torch.nn as nn
import torch
import pandas as pd

class NN_BN_Model(nn.Module):
    def __init__(self,hidden_layer=64):
        super(NN_BN_Model, self).__init__()
        self.inp = nn.Linear(5, hidden_layer)
        self.BN_inp = nn.BatchNorm1d(hidden_layer)
        self.linear = nn.Linear(hidden_layer,hidden_layer)
        self.BN_linear = nn.BatchNorm1d(hidden_layer)
        self.output = nn.Linear(hidden_layer,1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.BN_inp(self.inp(x)))
        x = self.relu(self.BN_linear(self.linear(x)))
        x = self.output(x)
        return x
    
class NN_Model(nn.Module):
    def __init__(self,hidden_layer):
        super(NN_Model, self).__init__()
        self.inp = nn.Linear(5, hidden_layer)
        self.linear = nn.Linear(hidden_layer,hidden_layer)
        self.output = nn.Linear(hidden_layer,1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.inp(x))
        x = self.relu(self.linear(x))
        x = self.output(x)
        return x
    
class NN():
    def __init__(self, model):
        self.device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
        df = pd.read_csv("data/log.csv")
        data = torch.from_numpy(df.values).float()
        x, y = data[:, :5].to(self.device), data[:, -1:].to(self.device)
        num_points = x.shape[0]
        self.train_x, self.valid_x = x[:int(num_points * 0.8)], x[int(num_points * 0.8):]
        self.train_y, self.valid_y = y[:int(num_points * 0.8)], y[int(num_points * 0.8):]
        self.model = model.to(self.device)
        self.training()

    def training(self):
        batch_size = 128
        training_step = 10000
        validation_interval = 1000
        lr = 1e-3

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss().to(self.device)
        self.train_losses = []
        self.valid_losses = []
        for step in range(training_step):
            idx = torch.randint(0, len(self.train_x), size=(batch_size, ))
            
            batch_train_x, batch_train_y = self.train_x[idx], self.train_y[idx]
            batch_pred_y = self.model(batch_train_x)

            optimizer.zero_grad()

            train_loss = loss_fn(batch_train_y, batch_pred_y)
            train_loss.backward()
            optimizer.step()

            self.train_losses.append(train_loss.item())

            self.model.eval()
            with torch.no_grad():
                valid_loss = loss_fn(self.model(self.valid_x), self.valid_y)
                self.valid_losses.append(valid_loss.item())
            self.model.train()

            if (step+1) % validation_interval == 0:
                print(f"Step: {step+1}/{training_step}\tTrain Loss: {self.train_losses[-1]:.2f}\tValid Loss: {self.valid_losses[-1]:.2f}")

        self.model.eval()
        self.model.to('cpu')

    def output(self, params):
        x = torch.tensor([params], dtype=torch.float32)
        output = self.model(x)
        return output.item()
    
    def best_loss(self):
        return min(self.valid_losses)