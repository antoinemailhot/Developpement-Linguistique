# Importation de librairies.
from PIL import Image # Importation de la biliothèque Pillow.
import numpy as np # Importation de la bibliothèque Numpy.
import csv

# Main
def main():
    chemin_repertoireEHWC = './EnglishHandwrittenCharacters/'
    chemin_fichier_csv = chemin_repertoireEHWC + 'english.csv'
    donnees = obtenir_tableau_traiter_depuis_fichier_csv(chemin_fichier_csv)
    for chemin_Image in donnees[:,0]:
        print(obtenir_tableau_par_image_png(chemin_repertoireEHWC + chemin_Image[0]))
            
# Fin du main.

# Calcul distance 

# Calcul distance hamming sur deux vecteur.
def distance_Hamming(vecteur1, vecteur2):
   return np.sum(vecteur1 != vecteur2)


# Fin calcul distance


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