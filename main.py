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
chemin_repertoireEHWC = "./EnglishHandwrittenCharacters/"
chemin_repertoire_Dictionnaires = "./Dictionnaires/"
# Pour seulement les chiffres : 55*10
# Pour seulement les chiffres + lettres majuscules : 55*36
# Pour tous charactères : 55*62
NOMBRE_IMAGES = 55*10
NOMBRES_VOISINS = 5

# Chargée un dictionnaire
def charger_dictionnaire(chemin_dictionnaire):
   with open(chemin_dictionnaire, "r", encoding="utf-8") as fichier:
      mots = {mot.strip() for mot in fichier}
   return mots

# Charger les dictionnaires.
dictionnaire_anglais = charger_dictionnaire(chemin_repertoire_Dictionnaires + "american-english")
dictionnaire_francais = charger_dictionnaire(chemin_repertoire_Dictionnaires + "french")

# Retourne un tableau Numpy à partir d'un fichier png.
def obtenir_tableau_par_image_png(chemin):
   # Ouvrir l'image avec Pillow.
   image = Image.open(chemin)

   # Convertit l'image en tableau.
   tableau_pixels = np.array(image)

   # Normaliser les valeurs de pixels entre 0 et 1.
   tableau_pixels = tableau_pixels[:,:,0] / 255

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
donnes_fichier_csv = obtenir_tableau_traiter_depuis_fichier_csv(chemin_repertoireEHWC + "english.csv")

# Chargements des images.
def charger_toutes_les_images (donnees):
   tableau_images = []
   chemins_images = [chemin_repertoireEHWC + str(chemin[0]).strip() for chemin in donnees[:, 0]]
   cpt=0
   for chemin in chemins_images:
      image = obtenir_tableau_par_image_png(chemin.strip())
      tableau_images.append(image)
      cpt+=1
      if(cpt>=NOMBRE_IMAGES):
         break
   return tableau_images

toutes_images = charger_toutes_les_images(donnes_fichier_csv)

# Valider un mot
def valider_mot(mot, dictionnaire):
   mot_minuscule = mot.lower()
   for mot_dictionnaire in dictionnaire:
      if(mot_dictionnaire.lower() == mot_minuscule):
         return True
   return False



# Main
def main():
    donnees = donnes_fichier_csv
    images = toutes_images #900 par 1200
    cpt = 0
    donnees_images = np.full(NOMBRE_IMAGES,None)
    for i in images:
       donnees_images[cpt] = [i, donnees[cpt][1][0]]
       cpt += 1
    k_plus_proches_voisins(donnees_images, fonction=1)
            
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
   return np.sum(np.abs(vecteur1-vecteur2))
   

# Mesure distance euclidienne. (p = 2) //INF1183-SE-09-data_minning.pdf p.20
def mesure_distance_euclidienne(vecteur1, vecteur2):
   return np.sqrt(np.sum(vecteur1 - vecteur2) ** 2)

def initialiser_position(donnes):
   positions = []     
   cpt_ligne = 0
   for valeur in donnes:
      for ligne in np.where(valeur == 0):
         if(len(ligne) != 0):
            for colonne in ligne:
               positions.append([cpt_ligne,colonne])
      cpt_ligne += 1
   return positions

#https://stackoverflow.com/questions/45742199/find-nearest-neighbors-of-a-numpy-array-in-list-of-numpy-arrays-using-euclidian
def nearest_neighbors(nn, donne):
   dists, idxs = nn.kneighbors(donne)
   return dists

# Fin calcul distance

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
def k_plus_proches_voisins(donnees, fonction):
   NB_IMAGES_PAR_CHARACTERE = 55
   POURCENTAGE_TEST = 0.1

   #indexs_test = indexs_aleatoires(POURCENTAGE_TEST, NB_IMAGES_PAR_CHARACTERE)
   indexs_test = [0,1,2,3,4]
   donnees, donnees_test = separer_donnees(donnees, indexs_test)

   if(fonction >= 1):
      for caractere_d in donnees:
         for d in caractere_d:
            d[0] = initialiser_position(d[0])
      for caractere_t in donnees_test:
         for t in caractere_t:
            t[0] = initialiser_position(t[0])


   with ThreadPoolExecutor(max_workers=5) as executeur:
      for caractere_t in donnees_test:

         # Utilisation de multi-thread : https://docs.python.org/3/library/concurrent.futures.html
         futures = []

         print("tests pour le caractère: ", caractere_t[0][1])
         for t in caractere_t:
            # Soumettre la tâche au ThreadPoolExecutor
            future = executeur.submit(trouver_voisins_proches, t, donnees, fonction)
            futures.append((t, future))

            # Attendre que toutes les futures soient terminées
         for t, future in futures:
               voisins = future.result()
               print(voisins)
               print(voter(voisins))
   return

def trouver_voisins_proches(test, donnees, fonction):
   voisins = [[None, None]] * NOMBRES_VOISINS
   distances_labels = []

   if(fonction == 1):
      nn = NearestNeighbors(n_neighbors=1, metric='manhattan', algorithm='auto')
      nn.fit(test[0])
   if(fonction == 2):
      nn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='auto')
      nn.fit(test[0])

   for caractere_d in donnees:
      for d in caractere_d:
         match fonction:
            case 0:distance = distance_Hamming(test[0], d[0])
            case 1:distance = np.sum(nearest_neighbors(nn, d[0]))
            case 2:distance = np.sum(nearest_neighbors(nn, d[0]))

         distances_labels.append((distance, d[1]))

   indices_k_plus_petites = np.argpartition([d[0] for d in distances_labels], NOMBRES_VOISINS)[:NOMBRES_VOISINS]

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

if __name__ == "__main__":
 main()