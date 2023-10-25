from PIL import Image
import os
from tqdm import tqdm

# Define the input and output directories
input_dir = '/Users/floriankainberger/Downloads/ASL_Alphabet_Dataset/asl_alphabet_train'
output_dir = 'data/train_resized4'

# Define the allowed class labels
# allowed_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
# ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
ALLOWED_LABELS = ['A', 'B', 'C', 'nothing']
MAX_COUNT_PER_CLASS = 900

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)


# Loop through the subdirectories (A, B, etc.)
for class_label in os.listdir(input_dir):
    if class_label in ALLOWED_LABELS:
        counter = 0
        input_class_dir = os.path.join(input_dir, class_label)
        output_class_dir = os.path.join(output_dir, class_label)

        os.makedirs(output_class_dir, exist_ok=True)

        # Get a list of image files in the class folder
        image_files = os.listdir(input_class_dir)

        # Create a progress bar
        progress_bar = tqdm(total=len(image_files), desc=f'Resizing {class_label}', unit='img')

        # Loop through the images in the class folder
        for image_name in image_files:
            counter+=1
            input_image_path = os.path.join(input_class_dir, image_name)
            output_image_path = os.path.join(output_class_dir, f"{image_name.split('.')[0]}_resized.jpg")

            # Open and resize the image
            with Image.open(input_image_path) as img:
                img_resized = img.resize((300, 300))  # Adjust dimensions as needed

                # Save the resized image
                img_resized.save(output_image_path)

            # Update the progress bar
            progress_bar.update(1)
            if counter >= MAX_COUNT_PER_CLASS:
                progress_bar.set_description(f'Reached the maximum number of images per class -> {MAX_COUNT_PER_CLASS}')
                break

        # Close the progress bar
        progress_bar.close()

print("Image resizing and saving complete.")