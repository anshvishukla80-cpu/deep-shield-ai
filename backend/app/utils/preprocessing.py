import torchvision.transforms as transforms

# Must match the training normalization AND Xception's 299×299 input size
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

def preprocess_image(image):
    return transform(image).unsqueeze(0)