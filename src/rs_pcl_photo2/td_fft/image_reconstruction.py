import numpy as np
import matplotlib.pyplot as plt

def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def calculate_distance_from_centre(coords, centre):
    # Distance from centre is √(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre) ** 2 + (coords[1] - centre) ** 2
        )
def find_symmetric_coordinates(coords, centre):
    return (centre + (centre - coords[0]),
            centre + (centre - coords[1])
            )

def square_image(image, array_size, centre):
    if image.ndim==3:
        image = image[:, :, :3].mean(axis=2)  # Convert to grayscale


    # Get all coordinate pairs in the left half of the array,
    # including the column at the centre of the array (which includes the centre pixel)
    coords_left_half = (
        (x, y) for x in range(array_size) for y in range(centre+1)
    )
    print("coord -> ", coords_left_half)
    # Sort points based on distance from centre
    return sorted(
        coords_left_half,
        key=lambda x: calculate_distance_from_centre(x, centre)
    )

def display_plots(individual_grating, reconstruction, idx):
    plt.subplot(121)
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)

