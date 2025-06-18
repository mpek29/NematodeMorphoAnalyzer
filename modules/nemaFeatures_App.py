# Importer les packages nécessaires
import cv2
import numpy as np
from skimage import morphology
from ultralytics import YOLO
from tkinter import Tk, Label, Button, filedialog, Menu, Canvas, Scrollbar, Frame, VERTICAL, HORIZONTAL
from PIL import Image, ImageTk


# Fonction pour obtenir l'échelle et les objets
def get_objects(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)#Seuillage automatique avec Otsu
    thresholds = cv2.bitwise_not(th)#Inversion des couleurs
    contours, _ = cv2.findContours(thresholds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)
    # Squelettisation
    skeleton = np.zeros_like(thresholds)
    cv2.drawContours(skeleton, [max_contour], -1, 255, 1)
    return skeleton ,max_contour
    
def get_scale(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresholds = cv2.bitwise_not(th)
    contours, _ = cv2.findContours(thresholds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if  w/h > 5:
            return 100/w

# Fonction pour la squelettisation
def skeletonization(obj):
    area = 0
    animal_image = None
    for i in range(len(obj)):
        contours, _ = cv2.findContours(obj[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            temp_area = cv2.contourArea(contours[0])
            if temp_area > area:
                area = temp_area
                animal_image = obj[i]
    if animal_image is None:
        raise ValueError("No suitable object found for skeletonization.")
    
    kernel = np.ones((12, 12), np.uint8)
    eroded_image = cv2.morphologyEx(animal_image, cv2.MORPH_CLOSE, kernel)
    skeleton = morphology.skeletonize(eroded_image.astype(np.uint8))
    return skeleton.astype(np.uint8)

# Fonction pour calculer la longueur
def calculate_length(obj_image, scale):
    skeleton = skeletonization(obj_image)
    skeleton_coords = np.argwhere(skeleton > 0)
    total_distance = 0
    for i in range(len(skeleton_coords) - 1):
        point1 = skeleton_coords[i]
        point2 = skeleton_coords[i+1]
        total_distance += np.linalg.norm(point1 - point2)
    length = total_distance * scale
    return length

# Fonction pour calculer la largeur
def calculate_width(obj_image, scale):

    image_gray = cv2.cvtColor(obj_image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    image = cv2.bitwise_not(th)

    skeleton = np.zeros(image.shape, np.uint8)
    image_copy = image.copy()
    iterations = 0
    while cv2.countNonZero(image_copy) > 0:
        temp = cv2.erode(image_copy, np.ones((3, 3), np.uint8), iterations=1)
        temp = cv2.dilate(temp, np.ones((3, 3), np.uint8), iterations=1)
        temp = cv2.subtract(image_copy, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image_copy = cv2.erode(image_copy, np.ones((3, 3), np.uint8), iterations=1)
        iterations += 1
    nb_pixel = 2 * iterations + 1
    width = nb_pixel * scale
   
    return width 


def remplissage_contour(th,max_contour, scale):
    length = 0
   
    total_distance = 0
    for i in range(len(max_contour) - 1):
        point1 = max_contour[i]
        point2 = max_contour[i+1]
        total_distance += np.linalg.norm(point1 - point2)
    length = total_distance * scale
    #remplissage
    im_floodfill = th.copy()
    img = th.copy()
    h, w = th.shape[:2]
    # Création de la mask pour le floodfill
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Remplissage de la région de l'image à partir du point (10, 10)
    cv2.floodFill(im_floodfill, mask, (10, 10), 255) 
    # Inverse de l'image après floodfill
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine l'image originale avec l'inverse du floodfill
    th = cv2.bitwise_or(th, im_floodfill_inv)
    # Squelettisation de l'image (extraction des contours)
    skeleton = np.zeros(th.shape, np.uint8)
    image_copy = th.copy()   
    iterations = 0
    while cv2.countNonZero(image_copy) > 0:
        if iterations > 10000:
            iterations=0
            break
         # rétrécir les régions blanches
        image_copy = cv2.erode(image_copy, np.ones((3, 3), np.uint8), iterations=1)
        iterations += 1
        if cv2.countNonZero(image_copy) == 0:
            break

    # Calcul de la largeur en fonction du nombre de pixels
    nb_pixel = 2 * iterations + 1
    width = nb_pixel * scale
    # Calcul de la longueur du squelette en parcourant les pixels
   
    return width, length/2,th

# Détection et calcul des dimensions pour chaque objet détecté
def main(imagePIL):
    # Charger l'image
    image = np.array(imagePIL)
    image_array = np.array(image)
    
    # Charger le modèle YOLO
    model = YOLO(r'models/task4_5_nemaFeatures/nema_features.pt')
    scale = get_scale(image)
    
    # Prédire les objets dans l'image
    results = model.predict(image)

    detected_obj = {}
    obj, maxc = get_objects(image)
    width_ele, length_ele, imageSkolette = remplissage_contour(obj, maxc, scale)

    # Initialiser les variables pour stocker les meilleures boîtes
    best_queue = None
    best_queue_conf = 0.0
    best_oesophagus = None
    best_oesophagus_conf = 0.0

    # Parcourir les résultats
    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for bbox, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x_min, y_min, x_max, y_max = map(int, bbox)
                cls_int = int(cls)

                # Trouver les boîtes ayant la meilleure confiance pour les classes 1 et 0
                if cls_int == 1 and conf > best_queue_conf:
                    best_queue_conf = conf
                    best_queue = (bbox, conf)
                elif cls_int == 0 and conf > best_oesophagus_conf:
                    best_oesophagus_conf = conf
                    best_oesophagus = (bbox, conf)

    # Dessiner et calculer les dimensions pour les boîtes avec la meilleure confiance
    if best_queue:
        bbox, conf = best_queue
        x_min, y_min, x_max, y_max = map(int, bbox)
        object_image = image_array[y_min:y_max, x_min:x_max]
        length = calculate_length(object_image, scale)
        width = calculate_width(object_image, scale)
        detected_obj["Queue"] = [width, length]
        # Dessiner la boîte
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, f"Queue ({conf:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if best_oesophagus:
        bbox, conf = best_oesophagus
        x_min, y_min, x_max, y_max = map(int, bbox)
        object_image = image_array[y_min:y_max, x_min:x_max]
        length = calculate_length(object_image, scale)
        width = calculate_width(object_image, scale)
        detected_obj["Oesophage"] = [width, length]
        # Dessiner la boîte
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, f"Oesophage ({conf:.2f})", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Afficher les résultats
    #print(f"Meiofaune: largeur = {width_ele:.2f} micromètre, longueur = {length_ele:.2f} micromètre")
    #for obj, dimensions in detected_obj.items():
        #print(f"{obj}: largeur = {dimensions[0]:.2f} micromètre, longueur = {dimensions[1]:.2f} micromètre")

    # Retourner les objets détectés et autres paramètres
    return detected_obj, width_ele, length_ele, image, imageSkolette





class ImageAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x800")
        self.root.title("Image Analyzer")

        # Initialisation des variables
        self.image_path = None
        self.image_panel = None
        self.skeleton_panel = None
        self.info_label = None
        self.info_Lbl_mesur = None

        # Style de l'application
        self.root.config(bg='lightblue')

        # Création du menu
        menu = Menu(self.root)
        self.root.config(menu=menu)
        file_menu = Menu(menu, tearoff=0)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Add Image", command=self.add_image)
        file_menu.add_command(label="Remove Image", command=self.remove_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Affichage initial : message centré
        self.info_label = Label(
            root,
            text="CHARGEZ UNE IMAGE",
            bg="lightblue",
            font=("Arial", 16, "bold"),
        )
        self.info_label.place(relx=0.5, rely=0.5, anchor="center")

    def add_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if self.image_path:
            # Appel de la fonction principale pour traiter l'image
            results, nematode_width, nematode_length, image, skel_image = main(self.image_path)

            # Texte des résultats
            #result_text = f"Meiofaune: Width = {nematode_width:.2f} μm, Length = {nematode_length:.2f} μm\n"
            #for obj, dimensions in results.items():
                #result_text += f"{obj.capitalize()}: Width = {dimensions[0]:.2f} μm, Length = {dimensions[1]:.2f} μm\n"

            # Mise à jour de l'interface
            self.info_label.place_forget()
            #self.display_image(image, skel_image, result_text)

    def remove_image(self):
        # Supprime les images et réinitialise l'affichage
        if self.image_panel:
            self.image_panel.destroy()
        if self.skeleton_panel:
            self.skeleton_panel.destroy()
        if self.info_Lbl_mesur:
            self.info_Lbl_mesur.destroy()

        # Réinitialisation du message centré
        self.info_label.config(text="CHARGEZ-VOUS UNE IMAGE POUR LA TRAITER")
        self.info_label.place(relx=0.5, rely=0.5, anchor="center")

    def display_image(self, image, skel_image, result_text):
        # Suppression des panneaux existants
        if self.image_panel:
            self.image_panel.destroy()
        if self.skeleton_panel:
            self.skeleton_panel.destroy()
        if self.info_Lbl_mesur:
            self.info_Lbl_mesur.destroy()

        # Conversion des images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((300, 300))
        image = ImageTk.PhotoImage(image)

        skel_image = Image.fromarray(skel_image)  # Multiplier par 255 pour l'affichage correct
        skel_image = skel_image.resize((300, 300))
        skel_image = ImageTk.PhotoImage(skel_image)

        # Affichage des images côte à côte
        self.image_panel = Label(self.root, image=image, bg="white", relief="groove",height=300 ,width=300)
        self.image_panel.image = image
        self.image_panel.pack(side="top", padx=10, pady=1)

        self.skeleton_panel = Label(self.root, image=skel_image, bg="white", relief="groove")
        self.skeleton_panel.image = skel_image
        self.skeleton_panel.pack(side="top", padx=10, pady=1)

        # Affichage des résultats en dessous des images
        self.info_Lbl_mesur = Label(
            self.root,
            text=result_text,
            bg="black",
            fg="white",
            font=("Arial", 14),
            relief="ridge",
            justify="center",
            anchor="center", 
        )
        self.info_Lbl_mesur.pack(fill="x", padx=10, pady=10)


if __name__ == "__main__":
    root = Tk()
    app = ImageAnalyzerApp(root)
    root.mainloop()



