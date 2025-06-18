import cv2
import pandas as pd
import os

csv_path = r"data\ground_truth.csv"
dossier_n1 = r"data\level_1"
dossier_n2 = r"data\level_2"
dossier_n3 = r"data\level_3"

df = pd.read_csv(csv_path)
index = 0
clicked = False
coords = (None, None)

def get_image_path(filename):
    path_n1 = os.path.join(dossier_n1, filename)
    path_n2 = os.path.join(dossier_n2, filename)
    path_n3 = os.path.join(dossier_n3, filename)
    if os.path.isfile(path_n2):
        return path_n2
    elif os.path.isfile(path_n3):
        return path_n3
    else:
        return None

def mouse_callback(event, x, y, flags, param):
    global clicked, coords
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked = True
        coords = (x, y)

def show_image():
    global index, df, clicked, coords

    while index < len(df):
        filename = df.loc[index, 'filename']
        img_path = get_image_path(filename)
        if img_path is None:
            print(f"Erreur: image {filename} non trouvée dans Niveau_2 ou Niveau_3.")
            index += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Erreur: impossible de charger {img_path}")
            index += 1
            continue

        clicked = False
        coords = (None, None)
        cv2.imshow('Image', img)
        cv2.setMouseCallback('Image', mouse_callback)

        print(f"Image {index+1}/{len(df)} : {filename}")
        print("Cliquez pour enregistrer coord, 'n' pour passer, 'q' pour quitter.")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if clicked:
                x, y = coords
                print(f"Coordonnées cliquées : ({x}, {y})")
                df.at[index, 'centerX'] = x
                df.at[index, 'centerY'] = y
                index += 1
                break
            elif key == ord('n'):  # passer sans clic
                print("Image ignorée, passage à la suivante.")
                index += 1
                break
            elif key == ord('q'):  # quitter
                print("Quitter demandé.")
                cv2.destroyAllWindows()
                df.to_csv(csv_path.replace('.csv', '_modified.csv'), index=False)
                print("Fichier CSV sauvegardé.")
                return

        cv2.destroyAllWindows()

    print("Toutes les images ont été traitées.")
    df.to_csv(csv_path.replace('.csv', '_modified.csv'), index=False)
    print("Fichier CSV sauvegardé.")

show_image()
