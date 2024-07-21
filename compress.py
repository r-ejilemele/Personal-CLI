from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def split_image(rgb_array):
    return (rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2])
def combine_channels(r,g,b):
    return np.stack((r, g, b), axis=-1)
def svd_compression(image_array, n_components=0):
    print("starting decomposition")
    U, s, Vh = np.linalg.svd(image_array, full_matrices=False)
    if n_components:
        s = s[:n_components]
        U = U[:, :n_components]
        Vh = Vh[:n_components, :]

    S = np.diag(s)

    # Reconstruct the matrix using np.dot
    reconstructed_dot = np.dot(U, np.dot(S, Vh))

    # Clip values to 0-255 and convert to uint8
    reconstructed_dot = np.clip(reconstructed_dot, 0, 255).astype(np.uint8)
    return reconstructed_dot
    
    
def compress(image_path):
    image = Image.open(image_path).convert('RGB')

    # Convert image to numpy array
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.show()

    r,g,b = split_image(image_array)
    channels = [0,0,0]
    channels = [0, 0, 0]
    for counter, array in enumerate((r, g, b)):
        channels[counter] = svd_compression(array, 50)
    combined = combine_channels(*channels)
    plt.imshow(combined)
    plt.show()

    
