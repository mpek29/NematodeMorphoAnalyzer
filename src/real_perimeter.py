import cv2
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import pour la barre de progression

# Fonction pour calculer le nombre de pixels utilisés pour un cercle avec contour
def calculer_pixels_cercle_contour(rayon, taille_image=2100):
    # Créer une image noire
    image = np.zeros((taille_image, taille_image, 3), dtype=np.uint8)

    # Paramètres du cercle
    centre = (taille_image // 2, taille_image // 2)  # Centre de l'image
    couleur = (0, 0, 255)  # Couleur du cercle (rouge en BGR)
    epaisseur = 1  # Épaisseur du contour (non rempli)

    # Dessiner un cercle avec contour (pas rempli)
    cv2.circle(image, centre, rayon, couleur, epaisseur)

    # Calculer le nombre de pixels rouges dans l'image (cercle)
    pixels_cercle = np.sum(np.all(image == couleur, axis=-1))
    
    return pixels_cercle

# Générer un tableau de résultats pour les rayons de 0 à 999
rayons = list(range(0, 1000))
resultats = []

# Utilisation de tqdm pour la barre de progression
for rayon in tqdm(rayons, desc="Calcul des pixels"):
    pixels = calculer_pixels_cercle_contour(rayon)
    perimetre_theorique = 2 * math.pi * rayon
    resultats.append([rayon, pixels, perimetre_theorique])

# Sauvegarder les résultats dans un fichier CSV
df_resultats = pd.DataFrame(resultats, columns=['Rayon', 'Pixels Utilisés', 'Périmètre Théorique'])
df_resultats.to_csv('data/pixels_par_rayon.csv', index=False)

print("Les résultats ont été sauvegardés dans 'pixels_par_rayon.csv'.")

# ----------- Ajout: Graphique -----------
plt.figure(figsize=(10, 6))
plt.plot(df_resultats['Rayon'], df_resultats['Pixels Utilisés'], label='Pixels Utilisés', color='red')
plt.plot(df_resultats['Rayon'], df_resultats['Périmètre Théorique'], label='Périmètre Théorique', color='blue')
plt.xlabel('Rayon')
plt.ylabel('Valeur')
plt.title('Pixels Utilisés et Périmètre Théorique en fonction du Rayon')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
