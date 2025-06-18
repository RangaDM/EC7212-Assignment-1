import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_intensity_levels(image, num_levels):
    """
    Reduce the number of intensity levels in a grayscale image.

    Args:
        image: Grayscale input image
        num_levels: Desired number of intensity levels (power of 2)

    Returns:
        Image with reduced intensity levels
    """
    # Step size for quantization
    step = 256 // num_levels

    # Create lookup table
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lookup_table[i] = (i // step) * step

    # Apply lookup table to image
    return cv2.LUT(image, lookup_table)

def main():
    # Load image
    image_path = input("Enter the path to your image: ")
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image.")
        return

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ask for intensity level
    num_levels = int(input("Enter number of intensity levels (power of 2, e.g., 2, 4, 8, 16, 32, 64, 128, 256): "))
    if num_levels not in [2, 4, 8, 16, 32, 64, 128, 256]:
        print("Invalid input! Must be a power of 2 between 2 and 256.")
        return

    # Process image
    reduced_image = reduce_intensity_levels(gray_image, num_levels)

    # Show results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale Image')

    plt.subplot(1, 2, 2)
    plt.imshow(reduced_image, cmap='gray')
    plt.title(f'Reduced to {num_levels} Levels')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
