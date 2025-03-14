import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

def load_image(image_path):
    """
    Load an image from a file path.
    Args:
        image_path (str): Path to the image file.
    Returns:
        PIL.Image: Loaded image.
    """
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess the image by resizing, normalizing, and converting to tensor.
    Args:
        image (PIL.Image): The input image to process.
    Returns:
        torch.Tensor: The preprocessed image as a tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),      # Resize image to 224x224 (standard for ResNet)
        transforms.ToTensor(),              # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for ResNet
    ])
    
    image_tensor = transform(image)
    return image_tensor

def augment_image(image):
    """
    Perform data augmentation to increase the variety of the dataset.
    Args:
        image (PIL.Image): The input image to augment.
    Returns:
        PIL.Image: The augmented image.
    """
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.RandomRotation(30),      # Random rotation between -30 and 30 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color adjustments
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2))  # Random affine transformations
    ])
    
    augmented_image = transform(image)
    return augmented_image

def save_image(image, save_path):
    """
    Save the image to a given path.
    Args:
        image (PIL.Image): The image to save.
        save_path (str): The destination file path to save the image.
    """
    try:
        image.save(save_path)
        print(f"Image saved to {save_path}")
    except Exception as e:
        print(f"Error saving image {save_path}: {e}")

def to_numpy(image_tensor):
    """
    Convert a PyTorch tensor to a numpy array.
    Args:
        image_tensor (torch.Tensor): The image tensor.
    Returns:
        np.array: The numpy array representation of the image.
    """
    image = image_tensor.permute(1, 2, 0).numpy()  # Change from (C, H, W) to (H, W, C)
    image = (image * 255).astype(np.uint8)  # Rescale from [0, 1] to [0, 255]
    return image

def load_and_preprocess(image_path):
    """
    Load and preprocess an image (resize, normalize, and convert to tensor).
    Args:
        image_path (str): Path to the image.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = load_image(image_path)
    if image is not None:
        image_tensor = preprocess_image(image)
        return image_tensor
    return None

def augment_and_preprocess(image_path):
    """
    Load, augment, and preprocess an image.
    Args:
        image_path (str): Path to the image.
    Returns:
        torch.Tensor: Augmented and preprocessed image tensor.
    """
    image = load_image(image_path)
    if image is not None:
        augmented_image = augment_image(image)
        image_tensor = preprocess_image(augmented_image)
        return image_tensor
    return None
