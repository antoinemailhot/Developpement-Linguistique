# But du projet : Pouvoir lire et reconnaître des charactères à partir d'une image.

# Importation de librairies.
from PIL import Image # Importation de la biliothèque Pillow.
import numpy as np # Importation de la bibliothèque Numpy.
import csv
import random
from concurrent.futures import ThreadPoolExecutor
from sklearn.neighbors import NearestNeighbors
from collections import Counter

#from scipy.spatial.distance import cdist

# Chargement des fichiers externes.

# Définir le chemin du répertoire EnglishHandwrittenCharacters
CHEMIN_REPERTOIRE_EHWC = "./EnglishHandwrittenCharacters/"
#CHEMIN_REPERTOIRE_DICTIONNAIRES = "./Dictionnaires/"

# Pour seulement les chiffres : 55*10
# Pour seulement les chiffres + lettres majuscules : 55*36
# Pour tous charactères : 55*62
NOMBRE_IMAGES = 55*10

# Retourne un tableau Numpy à partir d'un fichier png.
def obtenir_tableau_par_image_png(chemin):
   # Ouvrir l'image avec Pillow.
   image = Image.open(chemin)

   # Convertit l'image en tableau.
   tableau_pixels = np.array(image)

   # Normaliser les valeurs de pixels entre 0 et 1.
   tableau_pixels = tableau_pixels[:,:,0] // 255

   # Convertit l'image en un tableau et le retourne.
   return tableau_pixels

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

# Charger le fichier csv.
DONNEES_FICHIER_CSV = obtenir_tableau_traiter_depuis_fichier_csv(CHEMIN_REPERTOIRE_EHWC + "english.csv")

# Chargements des images.
def charger_toutes_les_images (donnees):
   tableau_images = []
   chemins_images = [CHEMIN_REPERTOIRE_EHWC + str(chemin[0]).strip() for chemin in donnees[:, 0]]
   cpt=0
   for chemin in chemins_images:
      image = obtenir_tableau_par_image_png(chemin.strip())
      tableau_images.append(image)
      cpt+=1
      if(cpt>=NOMBRE_IMAGES):
         break
   return tableau_images

TOUTES_IMAGES = charger_toutes_les_images(DONNEES_FICHIER_CSV)

# Main
def main():
    donnees = DONNEES_FICHIER_CSV
    images = TOUTES_IMAGES #900 par 1200
    cpt = 0
    donnees_images = np.full(NOMBRE_IMAGES,None)
    for i in images:
       donnees_images[cpt] = [i, donnees[cpt][1][0], None]
       cpt += 1
    
    entree_valide = False   
    while(not entree_valide): #Choix de la distance
      print("Choix de la distance (0)Hamming (1)Manhattan (2)Euclidienne")
      fonction = int(input())
      if(fonction == 0 or fonction == 1 or fonction == 2):
         entree_valide = True

    entree_valide = False
    while(not entree_valide): #Choix du k
      print("Choix du nombre de voisins (k)")
      k = int(input())
      if(k >= 1):
         entree_valide = True
   
    k_plus_proches_voisins(donnees_images, fonction, k)
            
# Fin du main.

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

def initialiser_position(donnes):
   NB_POSITIONS_EXCLUS = 2 #On garde la position d'un pixel sur NB_POSITIONS_EXCLUS au carré
   positions0 = []     
   positions1 = []  
   cpt_ligne = 0
   for ligne in donnes:         
      cpt_colonne = 0
      if((cpt_ligne % NB_POSITIONS_EXCLUS) == 0):
         for colonne in ligne:
            if((cpt_colonne % NB_POSITIONS_EXCLUS) == 0): 
               if (colonne == 0):
                  positions0.append([cpt_ligne,cpt_colonne])
               else:
                  positions1.append([cpt_ligne,cpt_colonne])
            cpt_colonne += 1   
      cpt_ligne += 1
   return positions0, positions1

#https://stackoverflow.com/questions/45742199/find-nearest-neighbors-of-a-numpy-array-in-list-of-numpy-arrays-using-euclidian
def nearest_neighbors(nn, donne):
   dists, idxs = nn.kneighbors(donne)
   return dists

#trouver le caractere le plus frequent dans les voisins
def voter(voisins):
    # Créer un compteur des caractères à partir des voisins
    compteur = Counter(v[1] for v in voisins)
    
    # Trouver le caractère le plus fréquent en utilisant most_common
    # most_common : https://docs.python.org/dev/library/collections.html#counter-objects
    caractere = compteur.most_common(1)[0][0]
    
    # Retourner le caractère le plus fréquent
    return caractere

# Calcul des K plus proches voisins.
def k_plus_proches_voisins(donnees, fonction, k):
   NB_IMAGES_PAR_CHARACTERE = 55
   POURCENTAGE_TEST = 0.1

   indexs_test = indexs_aleatoires(POURCENTAGE_TEST, NB_IMAGES_PAR_CHARACTERE)
   #indexs_test = [5,6,7,8,9] #indexs non aleatoire pour comparer les resultats des differentes distances
   donnees, donnees_test = separer_donnees(donnees, indexs_test)

   if(fonction >= 1):
      for caractere_d in donnees:
         for d in caractere_d:
            d[0],d[2] = initialiser_position(d[0])
      for caractere_t in donnees_test:
         for t in caractere_t:
            t[0],t[2] = initialiser_position(t[0])

   with ThreadPoolExecutor(max_workers=5) as executeur:
      for caractere_t in donnees_test:

         # Utilisation de multi-thread : https://docs.python.org/3/library/concurrent.futures.html
         futures = []

         print("tests pour le caractère: ", caractere_t[0][1])
         for t in caractere_t:
            # Soumettre la tâche au ThreadPoolExecutor
            future = executeur.submit(trouver_voisins_proches, t, donnees, fonction, k)
            futures.append((t, future))

            # Attendre que toutes les futures soient terminées
         for t, future in futures:
               voisins = future.result()
               print(voisins)
               print(voter(voisins))
   return

def trouver_voisins_proches(test, donnees, fonction, k):
   voisins = [[None, None]] * k
   distances_labels = []

   if(fonction == 1):
      nn0 = NearestNeighbors(n_neighbors=1, metric='manhattan', algorithm='auto')
      nn0.fit(test[0])
      nn1 = NearestNeighbors(n_neighbors=1, metric='manhattan', algorithm='auto')
      nn1.fit(test[2])
   if(fonction == 2):
      nn0 = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
      nn0.fit(test[0])
      nn1 = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
      nn1.fit(test[2])

   for caractere_d in donnees:
      for d in caractere_d:
         match fonction:
            case 0:distance = distance_Hamming(test[0], d[0])
            case 1:distance = np.sum(nearest_neighbors(nn0, d[0])) + np.sum(nearest_neighbors(nn1, d[2]))
            case 2:distance = np.sum(nearest_neighbors(nn0, d[0])) + np.sum(nearest_neighbors(nn1, d[2]))

         distances_labels.append((distance, d[1]))

   indices_k_plus_petites = np.argpartition([d[0] for d in distances_labels], k)[:k]

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
   NB_CARACTERES = int(NOMBRE_IMAGES/NB_IMAGES_PAR_CHARACTERE)
   index = 0
   index_t = 0
   index_d = 0

   tab_test = np.full((NB_CARACTERES, len(test)),None)
   tab_donnees = np.full((NB_CARACTERES, NB_IMAGES_PAR_CHARACTERE - len(test)),None)

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

# Valider un mot
#def valider_mot(mot, dictionnaire):
#   mot_minuscule = mot.lower()
#   for mot_dictionnaire in dictionnaire:
#      if(mot_dictionnaire.lower() == mot_minuscule):
#         return True
#   return False

 #Chargée un dictionnaire
#def charger_dictionnaire(chemin_dictionnaire):
#   with open(chemin_dictionnaire, "r", encoding="utf-8") as fichier:
#      mots = {mot.strip() for mot in fichier}
#   return mots

 #Charger les dictionnaires.
#DICTIONNAIRE_ANGLAIS = charger_dictionnaire(chemin_repertoire_Dictionnaires + "american-english")
#DICTIONNAIRE_FRANCAIS = charger_dictionnaire(chemin_repertoire_Dictionnaires + "french")

if __name__ == "__main__":
 main()