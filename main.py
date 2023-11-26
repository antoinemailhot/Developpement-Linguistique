# Importation de librairies.
from PIL import Image # Importation de la biliothèque Pillow.
import numpy as np # Importation de la bibliothèque Numpy.
import csv
import random

# Main
def main():
    chemin_repertoireEHWC = './EnglishHandwrittenCharacters/'
    chemin_fichier_csv = chemin_repertoireEHWC + 'english.csv'    
    VOISINS = 5
    donnees = obtenir_tableau_traiter_depuis_fichier_csv(chemin_fichier_csv)
    k_plus_proches_voisins(donnees, VOISINS)
    for chemin_Image in donnees[:,0]:
      print(obtenir_tableau_par_image_png(chemin_repertoireEHWC + chemin_Image[0]))
            
# Fin du main.

# Calcul distance 

# Calcul distance hamming sur deux vecteur. //Cours INF1183-SE-09-data_minning.pdf p.21
def distance_Hamming(vecteur1, vecteur2):
   return np.sum(vecteur1 != vecteur2)

# Calcul distance //Cours INF1183-SE-09-data_minning.pdf p.20

# Mesure distance manhattan. (p = 1) //INF1183-SE-09-data_minning.pdf p.20
def mesure_distance_manhattan(vecteur1, vecteur2):
   return np.sum(np.abs(vecteur1 - vecteur2))

# Mesure distance euclidienne. (p = 2) //INF1183-SE-09-data_minning.pdf p.20
def mesure_distance_euclidienne(vecteur1, vecteur2):
   return np.sqrt(np.sum(vecteur1 - vecteur2) ** 2)

# Fin calcul distance

# Calcul des K plus proches voisins.
def k_plus_proches_voisins(donnees, voisins):
   # TODO: A coder
   NB_IMAGES_PAR_CHARACTERE = 55
   POURCENTAGE_TEST = 0.1
   indexs_test = indexs_aleatoires(POURCENTAGE_TEST, NB_IMAGES_PAR_CHARACTERE)
   donnees, donnees_test = separer_donnees(donnees, indexs_test)

   return

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

   # Donne des indexs aléatoires
def separer_donnees(donnees, test):
   NB_IMAGES_PAR_CHARACTERE = 55
   NB_CARACTERES_DIFFERENTS = 62
   index = 0
   index_t = 0
   index_d = 0

   tab_test = [[None]*len(test)]*NB_CARACTERES_DIFFERENTS
   tab_donnees = [[None]*(NB_IMAGES_PAR_CHARACTERE - len(test))]*NB_CARACTERES_DIFFERENTS

   for d in donnees:
      if (test.count(index) > 0):
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