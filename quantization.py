import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats


import function as fun


def kmeans_Lloyd(X,centers_init,N,kmax=300):
    
    """
    Entrée:
        X: Echantillon de taille n x 1 suivant une densité f cible
        N: Nombre de quantificateur:
        kmax: nombre d'itération max
    
    Sortie:
        centers: liste finale des centroïdes
        lst_proba: liste des probabilités associées aux centres
        dist_tab: tableau de l'évolution de la distortion


    """


    # Initialisation pour éviter les problèmes d'affectation
    X_cible=np.copy(X) # On récupère la taille de l'échantillon
    centers=np.copy(centers_init)
    n=len(X_cible)
    dist_tab=[] # Récupération de l'évolution de la distortion

    for it in range(kmax):
    # Initialisation des variables courantes à l'itération it
        Assignation_cour=np.zeros(N)  # pour une version multidimensionelle on peut remplacer la ligne par " Assignation_cour=np.zeros((N,X_cible.shape[1]))"
        compteur_cour=np.zeros(N)
        dist_total_it=0

        for x in X_cible:
            indice,dist_x=fun.plusproche(x,centers)

            # Actualisation des variables courantes pour calculer le nouveau centre associé aux indices et de la distortion
            Assignation_cour[indice]=Assignation_cour[indice]+x
            compteur_cour[indice]+=1
            dist_total_it+=dist_x

        # Actualisation des centres
        for i in range(N):
            if(compteur_cour[i]>0):
                centers[i]=Assignation_cour[i]/compteur_cour[i]
            else:
                centers=centers

        dist_tab.append(dist_total_it/(n))

    # Calcul des probas de chaque cellule
    lst_proba=compteur_cour/n

    return centers, lst_proba, dist_tab



def grad_list(centers):    
    N=len(centers)
    grad=np.zeros(N)
    phi_plus=np.zeros(N)
    phi_moins=np.zeros(N)
    densi_plus=np.zeros(N)
    densi_moins=np.zeros(N)
    
    #On s'occupe d'abord des bords de la liste comme on actualise par rapport  au centre précedent et au suivant dans la liste 
    grad[0]=centers[0]*(scipy.stats.norm.cdf((centers[0]+centers[1])/2))+scipy.stats.norm.pdf((centers[0]+centers[1])/2)
    grad[N-1]=-centers[N-1]*scipy.stats.norm.cdf((centers[N-1]+centers[N-2])/2)-scipy.stats.norm.pdf((centers[N-1]+centers[N-2])/2)
    
    for i in range(1,N-1):
        phi_plus[i]=scipy.stats.norm.cdf((centers[i]+centers[i+1])/2)
        phi_moins[i]=scipy.stats.norm.cdf((centers[i]+centers[i-1])/2)
        densi_plus[i]=scipy.stats.norm.pdf((centers[i]+centers[i+1])/2)
        densi_moins[i]=scipy.stats.norm.pdf((centers[i]+centers[i-1])/2)
        
        
    grad=centers*(phi_plus-phi_moins)+densi_plus-densi_moins
    return grad



def method_Grad_normal(X,centers_init,N,gammainit,kmax=300):

    """"
    Entrée:
        X: Echantillon de taille n x 1 suivant une densité f cible
        N: Nombre de quantificateur:
        gammainit: valeur du pas
        kmax: nombre d'itération max
        
    Sortie:
        centers: liste finale des centroïdes ajustés à l'aide d'un gradient à pas déterministe
        lst_proba: liste des probabilités associées aux centres
        dist_tab: tableau de l'évolution de la distortion

    """
    # Initialisation pour éviter les problèmes d'affectation
    X_cible=np.copy(X)
    centers=np.copy(centers_init)
    n=len(X_cible)
    dist_cour=1
    dist_tab=[]
    compteur_cour=np.zeros(N)
    it=1
    
    while(it<kmax):
        compteur_cour=np.zeros(N)
        dist_cour=0
        gamma=gammainit
        centers=centers-gamma*grad_list(centers)
        
        for x in X_cible:
            indice,dist_x=fun.plusproche(x,centers)
            dist_cour=dist_cour+dist_x
            compteur_cour[indice]+=1
        dist_tab.append(dist_cour/n)
        it+=1   

    lst_proba=compteur_cour/n
    if(it!=kmax):
        print("itération max non atteinte !")
        
    return centers,lst_proba,dist_tab




def kmeans_Lloyd_d(X,centers_init,n,N,kmax=300):
    
    """
    Entrée:
        X: Echantillon de taille n x m suivant une densité f cible
        N: Nombre de quantificateurs:
        kmax: nombre d'itérations max
    
    Sortie:
        centers: liste finale des centroïdes selon l'algorithme de kmeans
        lst_proba: liste des probabilités associées aux centres
        dist_tab: Tableau de l'évolution de la distortion


    """


    # Initialisation pour éviter les problèmes d'affectation
    X_cible=np.copy(X) # On récupère la taille de l'échantillon
    centers=np.copy(centers_init) 
    

    dist_tab=[] # Récupération de l'évolution de la distortion

    for it in range(kmax):
    # Initialisation des variables courantes à l'itération it
        Assignation_cour=np.zeros((N,2)) # idem que pour centers
        compteur_cour=np.zeros(N)
        dist_total_it=0

        for x in X_cible:
            indice,dist_x=fun.plusproche_d(x,centers)

            # Actualisation des variables courante pour calculer le nouveau centre associé aux indices et de la distortion
            Assignation_cour[indice]=Assignation_cour[indice]+x
            compteur_cour[indice]+=1
            dist_total_it+=dist_x

        # Actualisation des centres
        for i in range(N):
            if(compteur_cour[i]>0):
                centers[i]=Assignation_cour[i]/compteur_cour[i]
            else:
                centers=centers

        dist_tab.append(dist_total_it/(2*n))

    # Calcul des probas de chaque cellule
    lst_proba=compteur_cour/n

    return centers, lst_proba, dist_tab


def gamma_n(it,N,gamma_0=0.01):
    a = 4.0 * N
    b = np.pi ** 2 / N 
    return gamma_0*a/(a+b*it)

def CLVQ_normal(X,centers_init,n,N,gamma_init,kmax=100):
    """
    
    Entrée:
        X: Echantillon de taille n x m suivant une densité f cible
        N: Nombre de quantificateurs:
        kmax: nombre d'itérations max
    
    Sortie: 
        centers: liste finale des centroïdes ajustée selon l'algorithme CLVQ
        lst_proba: liste des probabilités associées aux centres
        dist_tab: Tableau de l'évolution de la distortion

    """



    X_cible=np.copy(X) # On récupère la taille de l'échantillon
    centers=np.copy(centers_init)
    gamma=gamma_init

    dist_tab=[] # Récupération de l'évolution de la distortion

    for it in range(kmax):
        # Initialisation des variables courantes à l'itération it
        compteur_cour=np.zeros(N)
        dist_total_it=0

        for x in X_cible:
            #Phase de compétition
            indice,dist_x=fun.plusproche_d(x,centers)

            #Phase d'apprentissage            
            centers[indice]=centers[indice]-gamma_n(it,N,gamma_0=gamma)*(centers[indice]-x)
            
            # Actualisation des variables courante pour calculer le nouveau centre associé aux indices et de la distortion
            compteur_cour[indice]+=1
            dist_total_it+=dist_x

        dist_tab.append(dist_total_it/(2*n))
        
    # Calcul des proba de chaque cellules
    lst_proba=compteur_cour/n

    return centers, lst_proba, dist_tab
