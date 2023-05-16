import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr


def plusproche(p,lst):
    """ Summary: Nearest neighboor search function from a point to a 1D array

    Args:
        p (float): Point
        lst (array): _description_

    Returns:
        indice: index of nearest point of lst
        dist_min_cour_: minimum distance between p and the nearest point
    """
    dist_min_cour=float('inf') # initialisation de dist_cour arbitrairement grand pour simplifier la règle de décision
    indice=-1 


    for i,ci in enumerate(lst): # enumerate nous permet d'avoir accès au compteur et à l'objet itéré
        dist=np.abs(p-ci)**2 # Pour le cas muiltidimensionnel on remplace cette ligne par "dist=np.linalg.norm(p-ci)**2"
        if(dist<dist_min_cour):
            dist_min_cour=dist
            indice=i

    return indice,dist_min_cour
    
def plusproche_d(p,lst):
    """Summary: Nearest neighboor search function from a point to an array

    Args:
        p (float): Point
        lst (array): _description_

    Returns:
        indice: index of nearest point of lst
        dist_min_cour_: minimum distance between p and the nearest point
    """
    indice=-1 
    dist_tab = np.linalg.norm(lst - p, axis=1)
    indice = dist_tab.argmin()
    
    return indice, dist_tab[indice]

def affiche_voro_1D(centers):
    x = np.linspace(min(centers)-1, max(centers)+1, 100)
    arx = centers #points
    ary = [0]*len(arx) 
    
    plt.plot(x, len(x) * [0],color="black") #ligne bleue où se trouve les points
    plt.gca().axes.get_yaxis().set_visible(False) #pas de graduation sur y puis x
    plt.plot(arx[0], ary[0], marker="x", color="blue",label="Centroïde") #astuce pour afficher la légende 
    plt.plot(min(centers)-0.1,0, marker="|", color="black",label="Frontière")
    
    for i in range(1,len(centers)):
        plt.plot(arx[i], ary[i], marker="x", color="blue")   #affiche les points
        plt.plot((arx[i-1]+arx[i])/2,ary[i], marker="|", color="black")#frontière entre les cellules
    #legende axe x
    plt.legend()
    plt.xlim(min(centers)-0.1, max(centers)+0.1)
    plt.show()