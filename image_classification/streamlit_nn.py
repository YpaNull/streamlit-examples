import io
import os
import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Load the pre-trained model and labels
model = models.resnet18(pretrained=True)
labels_url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
labels_path = os.path.basename(labels_url)
if not os.path.exists(labels_path):
    labels = torch.hub.load_state_dict_from_url(labels_url)
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Create a function to preprocess the image and predict its label
def predict(image):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Predict the label
    with torch.no_grad():
        output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_label = torch.topk(probabilities, 1)
    label = labels[top_label[0]]

    return label, top_prob.item()

# Create the Streamlit app
def main():
    st.title('Image Classification with PyTorch')

    # Load the image
    image_file = st.file_uploader('Choose an image')
    if image_file is not None:
        image = Image.open(image_file)

        # Display the image and label prediction
        st.image(image, caption='Uploaded Image', use_column_width=True)
        label, prob = predict(image)
        st.write(f'Prediction: {label}, Probability: {prob:.2f}')

if __name__ == '__main__':
    main()
