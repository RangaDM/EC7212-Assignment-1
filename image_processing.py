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
    step = 256 // num_levels
    lookup_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lookup_table[i] = (i // step) * step
    return cv2.LUT(image, lookup_table)

def spatial_averaging(image, kernel_size):
    """
    Apply average blur to an image using the given kernel size.

    Args:
        image: Grayscale input image
        kernel_size: Size of the averaging kernel (e.g., 3 for 3x3)

    Returns:
        Blurred image
    """
    return cv2.blur(image, (kernel_size, kernel_size))

def main():
    # Load image
    image_path = input("Enter the path to your image: ")
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Unable to load the image.")
        return

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Part 1: Reduce intensity levels ---
    num_levels = int(input("Enter number of intensity levels (power of 2, e.g., 2, 4, 8...): "))
    if num_levels not in [2, 4, 8, 16, 32, 64, 128, 256]:
        print("Invalid input! Must be a power of 2 between 2 and 256.")
        return
    reduced_image = reduce_intensity_levels(gray_image, num_levels)

    # --- Part 2: Spatial averaging ---
    avg_3x3 = spatial_averaging(gray_image, 3)
    avg_10x10 = spatial_averaging(gray_image, 10)
    avg_20x20 = spatial_averaging(gray_image, 20)

    # Show results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Grayscale')

    plt.subplot(2, 3, 2)
    plt.imshow(reduced_image, cmap='gray')
    plt.title(f'Reduced to {num_levels} Levels')

    plt.subplot(2, 3, 4)
    plt.imshow(avg_3x3, cmap='gray')
    plt.title('3x3 Average Blur')

    plt.subplot(2, 3, 5)
    plt.imshow(avg_10x10, cmap='gray')
    plt.title('10x10 Average Blur')

    plt.subplot(2, 3, 6)
    plt.imshow(avg_20x20, cmap='gray')
    plt.title('20x20 Average Blur')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
