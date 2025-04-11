import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import pandas as pd

# Function to read labels from labels.txt
def read_labels_sequential(label_file):
    labels = []
    with open(label_file, 'r') as file:
        for line in file:
            label = line.strip()
            labels.append(label)
    return labels

# Path to your label.txt file
label_file = r"E:\breastcancer\labels.txt"

# Read labels into a list
labels = read_labels_sequential(label_file)

# Function to calculate GLCM properties
def calculate_glcm_properties(image):
    if len(image.shape) == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    image_gray = img_as_ubyte(gray)

    bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255])
    inds = np.digitize(image_gray, bins)

    max_value = inds.max() + 1
    matrix_cooccurrence = graycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=max_value, normed=False, symmetric=False)

    # Calculate GLCM properties
    contrast = graycoprops(matrix_cooccurrence, 'contrast')[0, 0]
    homogeneity = graycoprops(matrix_cooccurrence, 'homogeneity')[0, 0]
    energy = graycoprops(matrix_cooccurrence, 'energy')[0, 0]
    correlation = graycoprops(matrix_cooccurrence, 'correlation')[0, 0]

    return contrast, homogeneity, energy, correlation

# Process images and extract features
data_list = []
label_index = 0

#Path to your image folder and output CSV file
folder_path = r'E:\breastcancer\Output\CLAHE Images'

for filename in os.listdir(folder_path):
    if filename.endswith('.pgm'):
        img_path = os.path.join(folder_path, filename)
        img = io.imread(img_path)
        
        # Calculate GLCM properties
        contrast, homogeneity, energy, correlation = calculate_glcm_properties(img)
        
        # Get the label for the image
        label = labels[label_index]
        label_index += 1
        
        # Append the data to the list
        data_list.append([filename, f'[{contrast}, {correlation}, {energy}, {homogeneity}]', label])

# Convert list to pandas DataFrame
df = pd.DataFrame(data_list, columns=['Image Name', 'GLCM Properties', 'Label'])

# Save DataFrame to CSV file
df.to_csv('output_glcm_clahe.csv', index=False)
print("Done")