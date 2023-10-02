import matplotlib.pyplot as plt
import numpy as np
from module.trainingdata import generate_data_theo

# Affichage des résultats a1
def plots_results_a1(net, ax, i, nbrMoments, nbrAutoCov, num_samples=1000):
    inputs_tensor, targets_tensor = generate_data_theo(1000, nbrMoments = nbrMoments, nbrAutoCov = nbrAutoCov )
    inputs_tensor, targets_tensor = net(inputs_tensor).detach().numpy(), targets_tensor.detach().numpy()
    ax[i//3][i%3].scatter(targets_tensor[:,0],inputs_tensor[:,0], marker = '+', linestyle = 'None',c = targets_tensor[:,1])
    ax[i//3][i%3].plot((0,0.5),(0,0.5), color = 'red')
    ax[i//3][i%3].set_xlabel('target')
    ax[i//3][i%3].set_ylabel('output')
    ax[i//3][i%3].set_title('Etape : ' + str(i+1))
    ax[i//3][i%3].set_xlim([0,0.3])
    ax[i//3][i%3].set_ylim([0,0.3])


    