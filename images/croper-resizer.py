import cv2
import numpy as np
import os

def crop_image(input_image, left=145, top=270, width=280, height=210):
    # Check if the cropping coordinates are valid
    if left < 0 or top < 0 or width <= 0 or height <= 0 or left + width > input_image.shape[1] or top + height > input_image.shape[0]:
        print("Error: Invalid cropping coordinates!")
        return np.array([])  # Return an empty image in case of an error

    # Define the region of interest (ROI) for cropping
    roi = input_image[top:top + height, left:left + width]

    # Crop the image to the specified ROI
    cropped_image = roi.copy()  # Use .copy() to copy the data

    return cropped_image

def resize_image(input_image, target_width=427, target_height=240):
    # Check if the input image is not empty
    if input_image is None or input_image.size == 0:
        print("Error: Empty input image!")
        return np.array([])  # Return an empty image in case of an error

    # Resize the input image to the target size using bilinear interpolation
    resized_image = cv2.resize(input_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    return resized_image

def resize_with_aspect_ratio(image, target_width=288, target_height=216, color=(0, 0, 0)):
    # Get original dimensions
    h, w = image.shape[:2]

    # Compute scaling factors to maintain aspect ratio
    scale_w = target_width / w
    scale_h = target_height / h
    scale = min(scale_w, scale_h)

    # Calculate new size while keeping the aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize the image to fit within the target dimensions
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a new image with the target size and fill with the background color
    result = np.full((target_height, target_width, 3), color, dtype=np.uint8)

    # Calculate top-left corner to place the resized image
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2

    # Place the resized image on the center of the background
    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image

    return result

def process_images_in_directory():
    current_directory = os.getcwd()
    test_dir = os.path.join(current_directory, "test")
    train_dir = os.path.join(current_directory, "train")
    validation_dir = os.path.join(current_directory, "validation")

    test_files = os.listdir(test_dir)
    train_files = os.listdir(train_dir)
    validation_files = os.listdir(validation_dir)

    image_list = []
    
    # Construire la liste des images avec leur chemin complet
    for image in test_files:
        if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_list.append(os.path.join(test_dir, image))
    
    for image in train_files:
        if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_list.append(os.path.join(train_dir, image))
    
    for image in validation_files:
        if image.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_list.append(os.path.join(validation_dir, image))

    for image_path in image_list:
        input_image = cv2.imread(image_path)
        if input_image is None:
            print(f"Error: Cannot read image {image_path}")
            continue

        resized_image = resize_with_aspect_ratio(input_image)
        if resized_image.size == 0:
            print(f"Error: Resizing failed for image {image_path}")
            continue

        base_name, ext = os.path.splitext(os.path.basename(image_path))
        new_file_name = f"{base_name}_resized{ext}"
        output_path = os.path.join(os.path.dirname(image_path), new_file_name)
        cv2.imwrite(output_path, resized_image)
        print(f"Image {os.path.basename(image_path)} processed and saved as {new_file_name}")

if __name__ == "__main__":
    process_images_in_directory()
