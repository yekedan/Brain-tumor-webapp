import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes):
        super(BrainTumorCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32 * 56 * 56, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Loading the model 
def load_model(path='brain_tumor_model.pth'):
    model = BrainTumorCNN(num_classes=4)  
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Prediction function with confidence score
def predict(model, image_path, threshold=0.9):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        if confidence.item() < threshold:
            return -1, confidence.item()  # Indicate uncertainty if confidence is below threshold
    return predicted.item(), confidence.item()



# Writeup in English and French
def get_writeup(label):
    writeups = [
        {
            "en": "The image is classified as: No Tumor.",
            "fr": "L'image est classée comme: Pas de tumeur."
        },
        {
            "en": "The image is classified as: Glioma.",
            "fr": "L'image est classée comme: Gliome."
        },
        {
            "en": "The image is classified as: Meningioma.",
            "fr": "L'image est classée comme: Méningiome."
        },
        {
            "en": "The image is classified as: Pituitary Tumor.",
            "fr": "L'image est classée comme: Tumeur pituitaire."
        }
    ]
    return writeups[label]

    # Example usage
model = load_model()
prediction, confidence = predict(model, 'Te-gl_0011.jpg')
if prediction == -1:
    print("Model is uncertain about the prediction.")
else:
    writeup = get_writeup(prediction)
    print(writeup['en'])
    print(writeup['fr'])

