import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image

img_size = 224

def apply_clahe(img):
    # Convert the image to grayscale if it's not already
    if img.mode != 'L':
        img = img.convert('L')
    img_np = np.array(img)
    # Ensure that the image is of type CV_8UC1
    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_np)
    return Image.fromarray(img_clahe)

def apply_histogram_equalization(img):
    # Convert the PIL image to a numpy array
    img_np = np.array(img)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:  # Check if the image is RGB
        # Convert to YUV color space
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        # Apply histogram equalization on the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        # Convert back to RGB color space
        img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        # Apply histogram equalization on the grayscale image
        img_equalized = cv2.equalizeHist(img_np)
    # Convert back to PIL image
    return Image.fromarray(img_equalized)


def center_crop(img, crop_size):
    width, height = img.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = (width + crop_size) // 2
    bottom = (height + crop_size) // 2
    return img.crop((left, top, right, bottom))

transform_v1 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor()
])


transform_v2 = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Lambda(apply_clahe),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


transform_v3 = transforms.Compose([
    transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomRotation(degrees=20),
    transforms.Lambda(lambda img: apply_histogram_equalization(img)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])