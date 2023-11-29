# Importation de librairies.
from PIL import Image # Importation de la biliothèque Pillow.
import numpy as np # Importation de la bibliothèque Numpy.
import csv
import random
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cdist

# Main
def main():
    chemin_repertoireEHWC = './EnglishHandwrittenCharacters/'
    chemin_fichier_csv = chemin_repertoireEHWC + 'english.csv'    
    VOISINS = 5
    donnees = obtenir_tableau_traiter_depuis_fichier_csv(chemin_fichier_csv)
    k_plus_proches_voisins(donnees, VOISINS, chemin_repertoireEHWC)
    #for chemin_Image in donnees[:,0]:
      #print(obtenir_tableau_par_image_png(chemin_repertoireEHWC + chemin_Image[0]))
            
# Fin du main.

# Calcul distance 

# Calcul distance hamming sur deux vecteur. //Cours INF1183-SE-09-data_minning.pdf p.21
def distance_Hamming(vecteur1, vecteur2):
   #return np.sum(vecteur1 != vecteur2) Pour petit tableau
   # np.bitwise_xor : https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html
   # Calcul le xor au niveau du bit.
   # np.count_nonzero : https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html
   # Compte les valeurs non null.
   # Utilisez la fonction np.count_nonzero pour compter les bits différents
   # Faire attention la fonction bitwise_xor se crée un tableau temporaire donc si un trop gros tableau
   # cela peut générer une erreur.
    xor_result = np.bitwise_xor(vecteur1, vecteur2)
    return np.count_nonzero(xor_result)

# Calcul distance //Cours INF1183-SE-09-data_minning.pdf p.20

# Mesure distance manhattan. (p = 1) //INF1183-SE-09-data_minning.pdf p.20
def mesure_distance_manhattan(vecteur1, vecteur2):
   array1 = vecteur1[:, :, 0]
   array2 = vecteur2[:, :, 0]
   return cdist(array1, array2, metric='cityblock')
   

# Mesure distance euclidienne. (p = 2) //INF1183-SE-09-data_minning.pdf p.20
def mesure_distance_euclidienne(vecteur1, vecteur2):
   return np.sqrt(np.sum(vecteur1 - vecteur2) ** 2)

# Fin calcul distance

#trouver le caractere le plus frequent dans les voisins
from collections import Counter

def voter(voisins):
    # Créer un compteur des caractères à partir des voisins
    compteur = Counter(v[1] for v in voisins)
    
    # Trouver le caractère le plus fréquent en utilisant most_common
    # most_common : https://docs.python.org/dev/library/collections.html#counter-objects
    caractere = compteur.most_common(1)[0][0]
    
    # Retourner le caractère le plus fréquent
    return caractere




# Calcul des K plus proches voisins.
def k_plus_proches_voisins(donnees, nb_voisins, chemin_repertoireEHWC):
   # TODO: A coder
   NB_IMAGES_PAR_CHARACTERE = 55
   POURCENTAGE_TEST = 0.1
   chemin_fichier_csv = chemin_repertoireEHWC + 'english.csv'   

   indexs_test = indexs_aleatoires(POURCENTAGE_TEST, NB_IMAGES_PAR_CHARACTERE)
   donnees, donnees_test = separer_donnees(donnees, indexs_test)

   with ThreadPoolExecutor(max_workers=5) as executeur:
      for caractere_t in donnees_test:

         print("tests pour le caractère: ", caractere_t[0][1][0])

         # Utilisation de multi-thread : https://docs.python.org/3/library/concurrent.futures.html
         futures = []

         print("tests pour le caractère: ", caractere_t[0][1][0])
         for t in caractere_t:
            tableau_image_test = obtenir_tableau_par_image_png(chemin_repertoireEHWC + t[0][0])

            # Soumettre la tâche au ThreadPoolExecutor
            future = executeur.submit(trouver_voisins_proches, tableau_image_test, donnees, chemin_repertoireEHWC)
            print("Append future :")
            print(future)
            futures.append((t, future))

            # Attendre que toutes les futures soient terminées
         for t, future in futures:
               voisins = future.result()
               print(voter(voisins))



      #for t in caractere_t:
         #for caractere_d in donnees:
         #   for d in caractere_d:
         #      distance = distance_Hamming(tableau_image_test, obtenir_tableau_par_image_png(chemin_repertoireEHWC + d[0][0]))
         #      index_plus_grande_distance = 0
         #      i = 0
         #      while i < nb_voisins:
         #         # Si le voisin[i][0] est null alors on assigne l'index plus grande distance à i.
         #         if(voisins[i][0] is None):
         #            index_plus_grande_distance = i
         #            break
                  # Si le voisin[i][0] est plus grand que mon voisin le plus grand, alors il devient mon 
                  # voisin le plus grand.
         #         if(voisins[i][0] > voisins[index_plus_grande_distance][0]):
         #            index_plus_grande_distance = i

                  # on passe à l'index suivant.
         #         i += 1
         #      if(voisins[index_plus_grande_distance][0] is None or distance < voisins[index_plus_grande_distance][0]):
         #         voisins[index_plus_grande_distance] = [distance, d[1][0]]
         #print(voter(voisins))



   return

def trouver_voisins_proches(tableau_image_test, donnees, chemin_repertoireEHWC):
   nb_voisins = len(donnees[0])
   voisins = [[None, None]] * nb_voisins

   distances_labels = []

   for caractere_d in donnees:
      for d in caractere_d:
         distance = mesure_distance_manhattan(tableau_image_test, obtenir_tableau_par_image_png(chemin_repertoireEHWC + d[0][0]))
         distances_labels.append((distance, d[1][0]))

   indices_k_plus_petites = np.argpartition([d[0] for d in distances_labels], nb_voisins)[:nb_voisins]

   for i, index in enumerate(indices_k_plus_petites):
      voisins[i] = [distances_labels[index][0], distances_labels[index][1]]

   return voisins

# Donne des indexs aléatoires
def indexs_aleatoires(pourcentage:float, longueur:int):
   nbIndex = int(pourcentage * longueur)
   tabIndex = [None] * nbIndex
   for i in range(nbIndex):
      rand:int = random.randrange(0, longueur)
      j = 0
      while j <= i:
         if(tabIndex[j] == rand):
            j = 0
            if(rand == longueur-1):
               rand = 0
            else:
               rand += 1
         j += 1
      tabIndex[i] = rand
   return tabIndex

# sépare les données d'entrainement des données de test
def separer_donnees(donnees, test):
   NB_IMAGES_PAR_CHARACTERE = 55
   NB_CARACTERES_DIFFERENTS = 62
   index = 0
   index_t = 0
   index_d = 0

   tab_test = np.full((NB_CARACTERES_DIFFERENTS, len(test)),None)
   tab_donnees = np.full((NB_CARACTERES_DIFFERENTS, NB_IMAGES_PAR_CHARACTERE - len(test)),None)
   #tab_test = [[None]*len(test) for i in range(NB_CARACTERES_DIFFERENTS)]
   #tab_donnees = [[None]*(NB_IMAGES_PAR_CHARACTERE - len(test)) for i in range(NB_CARACTERES_DIFFERENTS)]

   for d in donnees:
      if (test.count(index % NB_IMAGES_PAR_CHARACTERE) > 0):
         tab_test[int(index/NB_IMAGES_PAR_CHARACTERE)][index_t] = d
         index_t += 1
         if(index_t == len(tab_test[0])):
            index_t = 0
      else:
         tab_donnees[int(index/NB_IMAGES_PAR_CHARACTERE)][index_d] = d
         index_d += 1
         if(index_d == len(tab_donnees[0])):
            index_d = 0
      index += 1

   return tab_donnees, tab_test

# Retourne un tableau Numpy à partir d'un fichier png.
def obtenir_tableau_par_image_png(chemin):
   # Ouvrir l'image avec Pillow.
   image = Image.open(chemin)

   # Convertit l'image en un tableau et le retourne.
   return np.array(image)

# Retourne les données du fichier csv dans un tableau.
def obtenir_tableau_depuis_fichier_csv(chemin):
   donnees = []
   with open(chemin, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for ligne in spamreader:
            donnees.append([element.split(',') for element in ligne])

   return np.array(donnees)

# Obtenir tableau traite du fichier csv.
def obtenir_tableau_traiter_depuis_fichier_csv(chemin):
   donnees = obtenir_tableau_depuis_fichier_csv(chemin)
   return donnees[1:]

if __name__ == "__main__":
 main()