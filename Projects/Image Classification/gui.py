from tkinter import *
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk
import os
import numpy as np
from tensorflow.keras.models import load_model
import PIL
import tensorflow_hub as hub

feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# Map prediction to names
number_to_name = {0: 'elon musk', 1: 'maria sharapova', 2: 'messi', 3: 'ronaldo', 4: 'virat'}
IMAGE_SHAPE = (224, 224)

hub_layer = hub.KerasLayer(
    feature_extractor_url,
    input_shape=IMAGE_SHAPE + (3,),
    trainable=False
)


def feature_extractor(x):
    return hub_layer(x)

# Load the model
model = load_model(r'Projects\Image Classification\image_classification_model.h5', custom_objects={
    'KerasLayer': hub.KerasLayer,
    'feature_extractor': feature_extractor
})



# Predict the image class
def predict_image(image_path):
    try:
        img = PIL.Image.open(image_path).convert("RGB").resize(IMAGE_SHAPE)
        img = np.array(img) / 255.0
        prediction = model.predict([img[np.newaxis, ...]])
        class_index = np.argmax(prediction)
        return number_to_name.get(class_index, "Unknown")
    except Exception as e:
        print("Prediction Error:", e)
        return "Prediction Failed"

# Handle dropped image
def handle_drop(event):
    paths = root.tk.splitlist(event.data)
    file_path = paths[0]

    if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
        show_image(file_path)
        predicted_name = predict_image(file_path)
        label.config(text=f"Prediction: {predicted_name}")
        print("Dropped file:", file_path)
    else:
        label.config(text="Not a supported image file.")

# Display image preview
def show_image(path):
    img = Image.open(path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Initialize GUI
root = TkinterDnD.Tk()
root.title("Drag & Drop Image Classifier")
root.geometry("500x400")

label = Label(root, text="Drop an image here", font=("Arial", 14))
label.pack(pady=20)

label.drop_target_register(DND_FILES)
label.dnd_bind('<<Drop>>', handle_drop)

image_label = Label(root)
image_label.pack(pady=20)

root.mainloop()
