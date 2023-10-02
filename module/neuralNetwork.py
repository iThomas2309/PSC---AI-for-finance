import torch
import torch.nn as nn

# Définition du réseau de neurones


class NeuralNetwork(nn.Module):
    def __init__(self,nbrMoments, nbrAutoCov, nbrNodes, nbrAutresParametres):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(nbrMoments + nbrAutoCov + nbrAutresParametres, nbrNodes)
        self.layer2 = nn.Linear(nbrNodes, nbrNodes)
        self.layer3 = nn.Linear(nbrNodes, nbrNodes)
        #self.layer4 = nn.Linear(nbrNodes, nbrNodes)
        self.layer5 = nn.Linear(nbrNodes, 2) # On a deux sorties : alpha_1 et beta_1

        self.layer1.bias.data.fill_(0)
        self.layer2.bias.data.fill_(0)
        self.layer3.bias.data.fill_(0)
        #self.layer4.bias.data.fill_(0)
        self.layer5.bias.data.fill_(0)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        #x = torch.relu(self.layer4(x))
        x = self.layer5(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))