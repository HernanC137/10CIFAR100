from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import io
import json
import os

# Configuración de dispositivo (GPU si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision import datasets

# Cargar el dataset CIFAR-100 para obtener las clases
trainset = datasets.CIFAR100(root='./data', train=True, download=True)
classes = trainset.classes 
# Cargar el modelo preentrenado ResNet101
model = models.resnet101(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 100)  # CIFAR-100 tiene 100 clases


model.load_state_dict(torch.load('cifar100.pkl', map_location=device))  #
model = model.to(device)
model.eval()  # Modo de evaluación para predicción

# Transformaciones para las imágenes de entrada
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])

# Crear la aplicación FastAPI
app = FastAPI()

# Ruta para predecir la clase de una imagen
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer la imagen cargada
        image = Image.open(io.BytesIO(await file.read()))
        image = transform(image).unsqueeze(0).to(device)

        # Realizar la predicción
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # Obtener el índice de la clase predicha
        predicted_class_idx = predicted.item()

        # Obtener el nombre de la clase predicha
        predicted_class_name = classes[predicted_class_idx]

        # Guardar predicción en un archivo JSON
        prediction_data = {"filename": file.filename, "predicted_class": predicted_class_name}
        save_prediction(prediction_data)

        # Devolver el nombre de la clase predicha
        return {"predicted_class": predicted_class_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {e}")


# Guardar la predicción en un archivo JSON
def save_prediction(prediction_data):
    file_name = 'predicciones.json'



    # Leer el archivo para agregar nuevas predicciones
    with open(file_name, 'r') as file:
        try:
            predictions = json.load(file)
        except json.JSONDecodeError:
            predictions = []

    # Agregar la nueva predicción
    predictions.append(prediction_data)

    # Guardar nuevamente el archivo con las predicciones
    with open(file_name, 'w') as file:
        json.dump(predictions, file, indent=4)

# Ejecutar la aplicación si se llama desde la terminal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
