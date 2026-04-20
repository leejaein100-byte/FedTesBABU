import os
import xml.etree.ElementTree as ET
from PIL import Image
import glob
from tqdm import tqdm
import shutil

def parse_bbox_from_annotation(annotation_path):
    """
    Parse bounding box from XML annotation file
    
    Args:
        annotation_path (str): Path to annotation XML file
        
    Returns:
        tuple: (xmin, ymin, xmax, ymax) or None if not found
    """
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        # Find bounding box
        bbox = root.find('object/bndbox')
        if bbox is not None:
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            return (xmin, ymin, xmax, ymax)
    except Exception as e:
        print(f"Error parsing annotation {annotation_path}: {e}")
    
    return None

def crop_image_with_bbox(image_path, bbox, padding_ratio=0.1):
    """
    Crop image using bounding box with optional padding
    
    Args:
        image_path (str): Path to input image
        bbox (tuple): (xmin, ymin, xmax, ymax)
        padding_ratio (float): Additional padding around bbox (0.1 = 10% padding)
    
    Returns:
        PIL.Image: Cropped image or None if error
    """
    try:
        image = Image.open(image_path).convert('RGB')
        
        if bbox is None:
            return image
        
        xmin, ymin, xmax, ymax = bbox
        width, height = image.size
        
        # Add padding
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        padding_x = int(bbox_width * padding_ratio)
        padding_y = int(bbox_height * padding_ratio)
        
        # Expand bbox with padding, ensuring we stay within image bounds
        xmin_padded = max(0, xmin - padding_x)
        ymin_padded = max(0, ymin - padding_y)
        xmax_padded = min(width, xmax + padding_x)
        ymax_padded = min(height, ymax + padding_y)
        
        # Crop the image
        cropped_image = image.crop((xmin_padded, ymin_padded, xmax_padded, ymax_padded))
        
        return cropped_image
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def crop_stanford_dogs_dataset(
    images_root='/home/jilee/jilee/data/Images',
    annotations_root='/home/jilee/jilee/data/Annotation', 
    output_root='/home/jilee/jilee/data/cropped images',
    padding_ratio=0.1,
    image_extensions=None
):
    """
    Crop all Stanford Dogs images using bounding box annotations
    
    Args:
        images_root (str): Root directory containing Images/breed_folders/
        annotations_root (str): Root directory containing Annotation/breed_folders/
        output_root (str): Output directory for cropped images
        padding_ratio (float): Padding around bounding box (0.1 = 10%)
        image_extensions (list): List of image extensions to process
    """
    if image_extensions is None:
        image_extensions = ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    # Verify input directories exist
    if not os.path.exists(images_root):
        raise FileNotFoundError(f"Images directory not found: {images_root}")
    
    if not os.path.exists(annotations_root):
        print(f"Warning: Annotations directory not found: {annotations_root}")
        print("Images will be copied without cropping.")
    
    # Get all breed folders
    breed_folders = [f for f in os.listdir(images_root) 
                    if os.path.isdir(os.path.join(images_root, f))]
    breed_folders.sort()
    
    print(f"Found {len(breed_folders)} breed folders")
    print(f"Processing images from: {images_root}")
    print(f"Using annotations from: {annotations_root}")
    print(f"Saving cropped images to: {output_root}")
    print(f"Padding ratio: {padding_ratio * 100}%")
    
    total_images = 0
    cropped_images = 0
    copied_images = 0
    failed_images = 0
    
    # Process each breed folder
    for breed_folder in tqdm(breed_folders, desc="Processing breeds"):
        breed_images_path = os.path.join(images_root, breed_folder)
        breed_annotations_path = os.path.join(annotations_root, breed_folder)
        breed_output_path = os.path.join(output_root, breed_folder)
        
        # Create output folder for this breed
        os.makedirs(breed_output_path, exist_ok=True)
        
        # Get all image files for this breed
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(breed_images_path, f"*.{ext}")
            image_files.extend(glob.glob(pattern))
        
        # Process each image
        for image_path in tqdm(image_files, desc=f"Processing {breed_folder}", leave=False):
            total_images += 1
            
            # Get image filename without extension
            image_filename = os.path.basename(image_path)
            image_name = os.path.splitext(image_filename)[0]
            
            # Construct annotation path
            annotation_path = os.path.join(breed_annotations_path, image_name)
            
            # Output path
            output_path = os.path.join(breed_output_path, image_filename)
            
            # Parse bounding box if annotation exists
            bbox = None
            if os.path.exists(annotation_path):
                bbox = parse_bbox_from_annotation(annotation_path)
            
            # Crop image
            cropped_image = crop_image_with_bbox(image_path, bbox, padding_ratio)
            
            if cropped_image is not None:
                try:
                    # Save cropped image
                    cropped_image.save(output_path, quality=95)
                    
                    if bbox is not None:
                        cropped_images += 1
                    else:
                        copied_images += 1
                        
                except Exception as e:
                    print(f"Error saving {output_path}: {e}")
                    failed_images += 1
            else:
                failed_images += 1
    
    # Print summary
    print(f"\n CROPPING SUMMARY:")
    print(f"{'='*50}")
    print(f"Total images processed: {total_images}")
    print(f"Successfully cropped (with bbox): {cropped_images}")
    print(f"Copied without cropping (no bbox): {copied_images}")
    print(f"Failed to process: {failed_images}")
    print(f"Success rate: {((cropped_images + copied_images) / total_images * 100):.1f}%")
    print(f"\nCropped images saved to: {output_root}")


if __name__ == "__main__":
    # Configuration
    IMAGES_ROOT = '/home/jilee/jilee/data/Images'
    ANNOTATIONS_ROOT = '/home/jilee/jilee/data/Annotation'
    OUTPUT_ROOT = '/home/jilee/jilee/data/cropped images'
    PADDING_RATIO = 0.1  # 10% padding around bounding box

    crop_stanford_dogs_dataset(
        images_root=IMAGES_ROOT,
        annotations_root=ANNOTATIONS_ROOT,
        output_root=OUTPUT_ROOT,
        padding_ratio=PADDING_RATIO)
