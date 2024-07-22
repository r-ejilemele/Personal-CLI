from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


def split_image(rgb_array):
    return (rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2])
def combine_channels(r,g,b):
    return np.stack((r, g, b), axis=-1)
def svd(image_array, quality=0):
    print("starting decomposition")
    U, s, Vh = np.linalg.svd(image_array, full_matrices=False)
    max_components = s.shape[0]
    if quality:
        quality = int(round(quality/100*max_components))
    else:
        quality = int(round(85/100*max_components))
    print(quality)
    s = s[:quality]
    U = U[:, :quality]
    Vh = Vh[:quality, :]
    

    S = np.diag(s)

    # Reconstruct the matrix using np.dot
    reconstructed_dot = np.dot(U, np.dot(S, Vh))

    # Clip values to 0-255 and convert to uint8
    reconstructed_dot = np.clip(reconstructed_dot, 0, 255).astype(np.uint8)
    return reconstructed_dot

def pca(image_matrix, quality):
    print("starting compression")
    mean = np.mean(image_matrix, axis=0)
    centered_matrix = image_matrix - mean

    covariance_matrix = np.cov(centered_matrix, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    if quality:
        quality = int(round(quality/100*len(sorted_eigenvalues)))
    else:
        quality = int(round(85/100*len(sorted_eigenvalues)))
    print(quality)
    selected_eigenvectors = sorted_eigenvectors[:, :quality]

    # Project the data
    projected_data = np.dot(centered_matrix, selected_eigenvectors)

    # Reconstruct the data
    reconstructed_dot =  np.dot(projected_data, selected_eigenvectors.T) + mean
    return np.clip(reconstructed_dot, 0, 255).astype(np.uint8)     

def pca_compress(image_path, quality=0):
    image = Image.open(image_path).convert('RGB')

    # Convert image to numpy array
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.show()

    r,g,b = split_image(image_array)
    channels = [0,0,0]
    channels = [0, 0, 0]
    for counter, array in enumerate((r, g, b)):
        channels[counter] = pca(array, quality)
    combined = combine_channels(*channels)
    image = Image.fromarray(combined)
    directory = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    image.save(os.path.join(directory, "compressed" + file_name), "JPEG")
    
def svd_compress(image_path, quality=0):
    image = Image.open(image_path).convert('RGB')

    # Convert image to numpy array
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.show()

    r,g,b = split_image(image_array)
    channels = [0,0,0]
    channels = [0, 0, 0]
    for counter, array in enumerate((r, g, b)):
        channels[counter] = svd(array, quality)
    combined = combine_channels(*channels)
    image = Image.fromarray(combined)
    directory = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    image.save(os.path.join(directory, "compressed" + file_name), "JPEG")

    
