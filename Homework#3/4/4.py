from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("--------------------------------------------------------------------------")
print('Using GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU available')
print("--------------------------------------------------------------------------\n")


#load data
housing_data = fetch_california_housing()
x, y = housing_data.data, housing_data.target


#Data description
print(housing_data.DESCR)
print(x.shape, y.shape)


#Split the train & test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


#Data normalization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


'''
1. transform your data to tensor (2 pts)
'''
x_train_tensor = torch.tensor(x_train, dtype=torch.float).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float).to(device)
x_test_tensor = torch.tensor(x_test, dtype=torch.float).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float).to(device)


train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

'''
2. Generate dataloader (3 pts)
'''
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


'''
3. Construct your own neural network model (5 pts)
'''
class NN(nn.Module):
  def __init__(self):
    super().__init__()

    self.model = nn.Sequential(
        nn.Linear(8, 128) , #input layer (8 features) -> hidden layer 1 (128 neurons)
        nn.ReLU(), #activation function for hidden layer 1
        nn.Linear(128, 64), #hidden layer 1 -> hidden layer 2 (64 neurons)
        nn.ReLU(), #activation function for hidden layer 2
        nn.Linear(64, 32), #hidden layer 2 -> hidden layer 3 (32 neurons)
        nn.ReLU(), #activation function for hidden layer 3
        nn.Linear(32, 1) #hidden layer 3 -> output layer (1 neuron)
        )

  def forward(self, x):
    return self.model(x)
  

'''
4. Define proper loss function and optimizer (5 pts)
'''
model = NN().to(device)
print(model)

loss = nn.MSELoss() #Loss function for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)


'''
5. Define the train function (3 pts)
'''
def train(model, trainloader, loss, optimizer, device, epochs = 100):
  model.train() #Start training
  for epoch in range(epochs):
    loss_history = 0

    for x, y in trainloader: #(batch 1_x, batch 1_y), (batch 2_x, batch 2_y), ...
      x, y = x.to(device), y.to(device)

      optimizer.zero_grad() #initialize the gradient

      y_hat = model(x) #forward propagation
      y_hat = y_hat.squeeze() #(64,)
      training_loss = loss(y_hat, y) #compute the loss of the minibatch

      training_loss.backward() #backpropagation
      optimizer.step()  #update the weights

      loss_history += training_loss.item()
    loss_history /= len(trainloader)
    print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1, epochs, loss_history))


'''
6. Define the test function - (Caution) you should use MSE loss to evaluate the model (3 pts)
'''
loss = nn.MSELoss() #Loss function for regression

def test(model, testloader, device):
  model.eval()    #model evaluation (weights are not updated)
  test_loss = 0
  total = 0

  with torch.no_grad():
    for x, y in testloader:
      x, y = x.to(device), y.to(device)

      y_hat = model(x) #prediction (64, 1)
      y_hat = y_hat.squeeze() #(64,)
      test_loss = test_loss + loss(y_hat, y).item()
      total += y.size(0)

  test_loss /= len(testloader)
  print('Test Loss: {:.4f}'.format(test_loss))


'''
7. Train your own model! (2 pts)
'''
train(model, train_loader, loss, optimizer, device, epochs = 30)

print("Train ", end="")
test(model, train_loader, device)   

'''
8. Test your own model! (2 pts)
'''
print("Test  ", end="")
test(model, test_loader, device)