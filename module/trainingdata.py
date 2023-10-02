# Génération des données d'entrainement
import numpy as np
import scipy.stats
import torch
from module.statistics import autocovariance, moments

import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def generate_coeff():
    # Générer des paramètres aléatoires pour a0, a1 et b1
    a0 = np.random.rand()*10**(-3) + 10**(-6)

    x1 = np.random.rand()
    x2 = np.random.rand()

    a1 = 0.3 - 0.3*max(x1,x2)
    b1 = min(x1,x2)
    return a0, a1, b1

def generate_data_theo(
        num_samples,
        nbrMoments = 3,
        nbrAutoCov = 3
    ):
    
    # Créer des listes pour stocker les données
    inputs = []
    targets = []

    # Générer les données d'entrainement
    for _ in range(num_samples):
        # Générer des paramètres aléatoires pour a0, a1 et b1

        #Intervalles dans lesquels les paramètres sont générés

        a0, a1, b1 = generate_coeff()
        
        # # Calculer les valeurs de sortie en utilisant la fonction f

        inputsSingle = moments(a0, a1, b1, nbrMoments)
        for i in range(nbrAutoCov):
            inputsSingle.append(autocovariance(a0, a1, b1, 2*i+2))
    

        # Ajouter les paramètres et les valeurs de sortie aux listes
        inputs.append(inputsSingle)
        targets.append([a1, b1])


    # Convertir les listes en tenseurs PyTorch
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)




def generate_data_test(
        a0,
        a1,
        b1,
        nbrMoments = 3,
        nbrAutoCov = 3
    ):
    
    # Créer des listes pour stocker les données
    inputs = []
    targets = []

    # Générer les données d'entrainement
        
    # # Calculer les valeurs de sortie
    inputsSingle = moments(a0, a1, b1, nbrMoments)
    inputsSingle.append((inputsSingle[1]-3)/inputsSingle[0]-1)
    for i in range(nbrAutoCov):
        inputsSingle.append(autocovariance(a0, a1, b1, 2*i+2))
            
    # Ajouter les paramètres et les valeurs de sortie aux listes
    inputs.append(inputsSingle)
    targets.append([a1, b1])

    # Convertir les listes en tenseurs PyTorch
    return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

def test():
    for i in range(100):
        a0, a1, b1 = generate_coeff()
        plt.scatter(a1,b1)
    plt.show()