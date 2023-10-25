from PIL import Image
import os
from tqdm import tqdm

# Define the input and output directories
input_dir = '/Users/floriankainberger/Downloads/ASL_Alphabet_Dataset/asl_alphabet_test'
output_dir = 'data/test_resized4'

# Define the allowed class labels
# allowed_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
# allowed_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
ALLOWED_LABELS = ['A', 'B', 'C', 'nothing']
# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

all_items = os.listdir(input_dir)

# Filter out directories and get only files
files = [item for item in all_items if os.path.isfile(os.path.join(input_dir, item))]
progress_bar = tqdm(total=len(files), desc=f'Resizing test images', unit='img')

for file in files:
    filename = os.path.splitext(os.path.basename(file))[0]

    className = filename.replace("_test", "")
    if className in ALLOWED_LABELS:
        os.makedirs(os.path.join(output_dir, className), exist_ok=True)

        input_image_path = os.path.join(input_dir, file)
        output_image_path = os.path.join(output_dir, className, f'{filename}_resized.jpg')

        with Image.open(input_image_path) as img:
            img_resized = img.resize((300, 300))  # Adjust dimensions as needed

            # Save the resized image
            img_resized.save(output_image_path)
    
    progress_bar.update(1)

progress_bar.close()
