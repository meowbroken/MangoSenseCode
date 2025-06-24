import tensorflow as tf
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, Label, Button
import gc

MODEL_PATH = 'models/mango-fruit-model.keras' 
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)
class_names = ['Alternaria', 'Anthracnose', 'Black Mould Rot', 'Healthy', 'Stem end Rot']

def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img.close()  # Ensure image is closed after use
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    
def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        print("Selected file:", file_path)
        img = Image.open(file_path)
        img.show()
        img.close()  # Close after showing
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        print("Prediction vector:", prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100
        print(f"Predicted class: {predicted_class} ({confidence:.2f}%)")
        result_label.config(text=f"Prediction: {predicted_class} ({confidence:.2f}%)")
        gc.collect()

root = tk.Tk()
root.title("Mango Disease/Insect Detection")

upload_btn = Button(root, text="Upload Image", command=predict_image)
upload_btn.pack(pady=20)

result_label = Label(root, text="Upload an image to get prediction")
result_label.pack(pady=20)

root.mainloop()