import torch

# Define the model
class ANN(torch.nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = torch.nn.Linear(377, 2048)
        self.fc2 = torch.nn.Linear(2048, 4096)
        self.fc3 = torch.nn.Linear(4096, 2048)
        self.fc4 = torch.nn.Linear(2048,1024)
        self.fc5 = torch.nn.Linear(1024, 773)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x