import numpy as np
import scipy.stats
import torch


from module.trainingdata import generate_coeff

# Génération de signaux

def signal(n, a0, a1, b1):
    signalTab = np.zeros(n)
    signalTab[0] = np.sqrt(a0/(1-a1-b1)) * np.random.normal(1, 1)
    sigma = a0
    signalTab[0] = a0 * np.random.normal(0, 1)
    for i in range(1, n):
        sigma = np.sqrt(a0 + a1 * signalTab[i-1]**2 + b1 * sigma**2)
        signalTab[i] = sigma * np.random.normal(0, 1)
    return signalTab



# Génération de données à partir de signaux

def generate_data(
          num_inputs,  # Nombre de données à générer
          num_samples, # Nombre de points par données
          time_sample, # Temps de la série
          nbrMoments = 3, # Nombre de moments à calculer
          nbrAutoCov = 3, # Nombre d'autocovariances à calculer
          isCoeffRandom = False,
          a0 = 0.0001,
          a1 = 0.2,
          b1 = 0.4
        ):

        inputs = []
        targets = []

        for _ in range(num_inputs):
            # Créer des listes pour stocker les données
            yn = []
            inputsSingle = []
            # Gestion des paramètres
            
            if isCoeffRandom:
                a0, a1, b1 = generate_coeff()
            
            #yn = signal(time_sample, a0, a1, b1)
            # Générer les données d'entrainement
            for _ in range(num_samples):
                # Générer un signal GARCH(1,1)
                y = signal(time_sample, a0, a1, b1)
                yn.append(y[time_sample-1])

            # Calcule les moments d'odre 2,4 et 6 de yn
            var = scipy.stats.moment(yn, moment=2,center=0) 
            for i in range(nbrMoments):
                inputsSingle.append(scipy.stats.moment(yn, moment=2*i+2,center = 0))

            # Normalisation des moments
            for i in range(1,nbrMoments):
                 inputsSingle[i] /= var**(i+1)

            # Calcule les autocovariances d'ordre 2,4 et 6 de yn
            for i in range(1,nbrAutoCov+1):
                inputsSingle.append(np.cov(yn[0:-2*i],yn[2*i:time_sample])[0][1]/var)

            
            inputs.append(inputsSingle)
            targets.append([a1])

        # Convertir les listes en tenseurs PyTorch
        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)