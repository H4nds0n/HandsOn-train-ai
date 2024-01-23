from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import math

# Define the input and output directories
input_dir = '/Users/floriankainberger/Downloads/ASL_Alphabet_Dataset/asl_alphabet_train'
output_dir = 'data/test_resized60'

# Define the allowed class labels
# allowed_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
# ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'F', 'H', 'K', 'L', 'O', 'U']
# ALLOWED_LABELS = ['A', 'B', 'C', 'D', 'K']
MAX_COUNT_PER_CLASS = 1000
SIZE = 224

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
        progress_bar = tqdm(total=MAX_COUNT_PER_CLASS, desc=f'Resizing {class_label}', unit='img')

        # Loop through the images in the class folder
        for image_name in image_files:
            counter+=1
            input_image_path = os.path.join(input_class_dir, image_name)
            output_image_path = os.path.join(output_class_dir, f"{image_name.split('.')[0]}_resized.jpg")

            # Open and resize the image
            with Image.open(input_image_path) as img:
                parentImg = np.ones((SIZE,SIZE,3), np.uint8)*255
                height, width = img.size
                aspectRatio = height/width
                
                newH = height
                newW = width

                wGap = 0
                hGap = 0

                if aspectRatio > 1:
                    newW = math.floor(SIZE/aspectRatio)
                    newH = SIZE
                    wGap = wGap = math.floor((SIZE-newW)/2)
                else:
                    newH = math.floor(SIZE*aspectRatio)
                    newW = SIZE
                    hGap = math.floor((SIZE-newH)/2)
                
                resizedImg = img.resize((newW, newH))

                parentImg[hGap:newH+hGap, wGap:newW+wGap] = resizedImg

                
                result_img = Image.fromarray(parentImg)

                result_img.save(output_image_path)
                # img_resized = img.resize((300, 300))  # Adjust dimensions as needed

                # Save the resized image
                # img_resized.save(output_image_path)

            # Update the progress bar
            progress_bar.update(1)
            if counter >= MAX_COUNT_PER_CLASS:
                progress_bar.set_description(f'Reached the maximum number of images per class -> {MAX_COUNT_PER_CLASS}')
                break

        # Close the progress bar
        progress_bar.close()

print("Image resizing and saving complete.")

# from PIL import Image
# import os
# from tqdm import tqdm

# # Define the input and output directories
# input_dir = '/Users/floriankainberger/Downloads/ASL_Alphabet_Dataset/asl_alphabet_test'
# output_dir = 'data/test_resized1'

# # Define the allowed class labels
# # allowed_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
# # allowed_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
# ALLOWED_LABELS = ['A', 'B', 'C', 'D']
# # Create the output directory if it doesn't exist
# os.makedirs(output_dir, exist_ok=True)

# all_items = os.listdir(input_dir)

# # Filter out directories and get only files
# files = [item for item in all_items if os.path.isfile(os.path.join(input_dir, item))]
# progress_bar = tqdm(total=len(files), desc=f'Resizing test images', unit='img')

# for file in files:
#     filename = os.path.splitext(os.path.basename(file))[0]

#     className = filename.replace("_test", "")
#     if className in ALLOWED_LABELS:
#         os.makedirs(os.path.join(output_dir, className), exist_ok=True)

#         input_image_path = os.path.join(input_dir, file)
#         output_image_path = os.path.join(output_dir, className, f'{filename}_resized.jpg')

#         with Image.open(input_image_path) as img:
#             img_resized = img.resize((300, 300))  # Adjust dimensions as needed

#             # Save the resized image
#             img_resized.save(output_image_path)
    
#     progress_bar.update(1)

# progress_bar.close()
