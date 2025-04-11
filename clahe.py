import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def adaptive_histogram_equalization(image_path,
                                    clip_limit=2.0,
                                    tile_grid_size=(8, 8)):
  # Read the PGM image
  image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

  if image is None:
    raise FileNotFoundError(f"The image at path {image_path} was not found.")

  # Create a CLAHE object (Arguments are optional).
  clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

  # Apply CLAHE to the grayscale image
  cl1 = clahe.apply(image)

  return image, cl1


def plot_results(original, equalized):
  # Plot the original and equalized images
  plt.figure(figsize=(10, 5))

  plt.subplot(1, 2, 1)
  plt.title('Original Image')
  plt.imshow(original, cmap='gray')
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.title('Adaptive Histogram Equalized Image')
  plt.imshow(equalized, cmap='gray')
  plt.axis('off')

  plt.show()


def process_dataset(input_directory,
                    output_directory,
                    clip_limit=2.0,
                    tile_grid_size=(8, 8)):
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)

  for filename in os.listdir(input_directory):
    if filename.endswith(".pgm"):  # Process only PGM files
      input_path = os.path.join(input_directory, filename)
      output_path = os.path.join(output_directory, filename)

      try:
        original_image, equalized_image = adaptive_histogram_equalization(
            input_path, clip_limit, tile_grid_size)
        cv2.imwrite(output_path, equalized_image)

        print(f"Processed and saved: {filename}")

      except FileNotFoundError as e:
        print(e)


if __name__ == "__main__":
  # Paths to the input and output directories
  input_directory = r"E:\breastcancer\MIAS-Dataset"
  output_directory = r"E:\breastcancer\Output\CLAHE Images"

  # Process the entire dataset
  process_dataset(input_directory, output_directory)
