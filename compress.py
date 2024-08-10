from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fftpack import dct, idct

quant_dict = {
        8: np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ]
        ),
        12: np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61, 62, 63, 64, 65],
                [12, 12, 14, 19, 26, 58, 60, 55, 56, 57, 58, 59],
                [14, 13, 16, 24, 40, 57, 69, 56, 57, 58, 59, 60],
                [14, 17, 22, 29, 51, 87, 80, 62, 63, 64, 65, 66],
                [18, 22, 37, 56, 68, 109, 103, 77, 78, 79, 80, 81],
                [24, 35, 55, 64, 81, 104, 113, 92, 93, 94, 95, 96],
                [49, 64, 78, 87, 103, 121, 120, 101, 102, 103, 104, 105],
                [72, 92, 95, 98, 112, 100, 103, 99, 100, 101, 102, 103],
                [75, 95, 97, 99, 113, 101, 105, 101, 102, 103, 104, 105],
                [78, 98, 100, 102, 115, 103, 106, 103, 104, 105, 106, 107],
                [80, 100, 102, 104, 116, 105, 108, 105, 106, 107, 108, 109],
                [82, 102, 104, 106, 118, 107, 110, 107, 108, 109, 110, 111],
            ]
        ),
        16: np.array(
            [
                [16, 11, 10, 16, 24, 40, 51, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                [12, 12, 14, 19, 26, 58, 60, 55, 56, 57, 58, 59, 60, 61, 62, 63],
                [14, 13, 16, 24, 40, 57, 69, 56, 57, 58, 59, 60, 61, 62, 63, 64],
                [14, 17, 22, 29, 51, 87, 80, 62, 63, 64, 65, 66, 67, 68, 69, 70],
                [18, 22, 37, 56, 68, 109, 103, 77, 78, 79, 80, 81, 82, 83, 84, 85],
                [24, 35, 55, 64, 81, 104, 113, 92, 93, 94, 95, 96, 97, 98, 99, 100],
                [
                    49,
                    64,
                    78,
                    87,
                    103,
                    121,
                    120,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                ],
                [
                    72,
                    92,
                    95,
                    98,
                    112,
                    100,
                    103,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                ],
                [
                    75,
                    95,
                    97,
                    99,
                    113,
                    101,
                    105,
                    101,
                    102,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                ],
                [
                    78,
                    98,
                    100,
                    102,
                    115,
                    103,
                    106,
                    103,
                    104,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                ],
                [
                    80,
                    100,
                    102,
                    104,
                    116,
                    105,
                    108,
                    105,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                ],
                [
                    82,
                    102,
                    104,
                    106,
                    118,
                    107,
                    110,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                ],
                [
                    84,
                    104,
                    106,
                    108,
                    120,
                    109,
                    112,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                ],
                [
                    86,
                    106,
                    108,
                    110,
                    122,
                    111,
                    114,
                    111,
                    112,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                ],
                [
                    88,
                    108,
                    110,
                    112,
                    124,
                    113,
                    116,
                    113,
                    114,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                ],
                [
                    90,
                    110,
                    112,
                    114,
                    126,
                    115,
                    118,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    123,
                ],
            ]
        ),
    }
def split_image(rgb_array):
    return (rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2])


def combine_channels(r, g, b):
    return np.stack((r, g, b), axis=-1)


def svd(image_array, quality=0):
    print("starting decomposition")
    U, s, Vh = np.linalg.svd(image_array, full_matrices=False)
    max_components = s.shape[0]
    if quality:
        quality = int(round(quality / 100 * max_components))
    else:
        quality = int(round(85 / 100 * max_components))
    print(quality)
    s = s[:quality]
    U = U[:, :quality]
    Vh = Vh[:quality, :]

    S = np.diag(s)

    # return (U, S, Vh)

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
        quality = int(round(quality / 100 * len(sorted_eigenvalues)))
    else:
        quality = int(round(85 / 100 * len(sorted_eigenvalues)))
    print(quality)
    selected_eigenvectors = sorted_eigenvectors[:, :quality]

    # Project the data
    projected_data = np.dot(centered_matrix, selected_eigenvectors)

    # return (projected_data, selected_eigenvectors, mean)
    # Reconstruct the data
    reconstructed_dot = np.dot(projected_data, selected_eigenvectors.T) + mean
    return np.clip(reconstructed_dot, 0, 255).astype(np.uint8)


# Step 1: Convert RGB to YCbCr
def rgb_to_ycbcr(image):
    return image.convert("YCbCr")


# Step 4: Quantization using a standard quantization matrix
quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)


def block_average(arr, block_size):
    h, w = arr.shape
    new_h, new_w = h // block_size, w // block_size
    if h % block_size != 0 or w % block_size != 0:
        # Adjust the shape of avg_arr to accommodate all blocks
        new_h += 1
        new_w += 1
    avg_arr = np.zeros((new_h, new_w), dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = arr[i : i + block_size, j : j + block_size]
            avg_arr[i // block_size, j // block_size] = np.mean(block)
    return avg_arr


def stretch_array(arr, target_shape):
    from scipy.ndimage import zoom

    return zoom(
        arr, [target_shape[0] / arr.shape[0], target_shape[1] / arr.shape[1]], order=3
    )


def block_division(image):
    h, w = image.shape
    return [image[i : i + 8, j : j + 8] for i in range(0, h, 8) for j in range(0, w, 8)]


def apply_dct(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def apply_idct(block):
    return idct(idct(block.T, norm="ortho").T, norm="ortho")


def pad_block(block, block_size):
    h, w = block.shape
    padded_block = np.zeros((block_size, block_size), dtype=block.dtype)
    padded_block[:h, :w] = block
    return padded_block


def quantize(block, quant_matrix):
    if quant_matrix.shape != block.shape:
        quant_matrix = np.resize(quant_matrix, block.shape)
    return np.round(block / quant_matrix)


def dequantize(block, quant_matrix):
    if quant_matrix.shape != block.shape:
        quant_matrix = np.resize(quant_matrix, block.shape)
    return block * quant_matrix


def rle(matrix):
    # Find the indices where the value changes
    change_indices = np.where(matrix[:-1] != matrix[1:])[0] + 1

    # Add start and end indices
    change_indices = np.concatenate(([0], change_indices, [len(matrix)]))

    # Calculate the run lengths and values
    run_lengths = np.diff(change_indices)
    values = matrix[change_indices[:-1]]

    # Combine values and run lengths
    return np.column_stack((values, run_lengths))


def undo_rle(matrix):
    values = matrix[:, 0].astype(int)
    run_lengths = matrix[:, 1].astype(int)

    # Reconstruct the 1D array from RLE
    return np.repeat(values, run_lengths)


def dct_function(matrix, block_size):
    dct_matrix = np.zeros_like(matrix)
    # block_size = 8
    for i in range(0, matrix.shape[0], block_size):
        for j in range(0, matrix.shape[1], block_size):
            block = matrix[i : i + block_size, j : j + block_size]
            dct_block = apply_dct(block)
            quantized_block = quantize(dct_block, quant_dict[block_size])
            dct_matrix[i : i + block_size, j : j + block_size] = quantized_block
    return dct_matrix.astype("float32")


def undo_dct_function(matrix, block_size, shape):
    height, width = shape
    i_dct = np.zeros_like(matrix)
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            quantized_block = matrix[i : i + block_size, j : j + block_size]
            dequantized_block = dequantize(quantized_block, quant_dict[block_size])
            idct_block = apply_idct(dequantized_block)
            i_dct[i : i + block_size, j : j + block_size] = idct_block
    return i_dct


# JPEG Compression Pipeline
def dct_compress(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")
    image = rgb_to_ycbcr(image)
    y, cb, cr = image.split()

    # Convert to NumPy arrays
    y = np.array(y, dtype=np.float32) - 128
    cb = np.array(cb, dtype=np.float32) - 128
    cr = np.array(cr, dtype=np.float32) - 128

    # Shrink Cb and Cr arrays
    cb_shrinked = block_average(cb, 8)
    cr_shrinked = block_average(cr, 8)

    
    # Define quantization matrix
    quantization_matrix = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )
    # Perform DCT on Y array and quantize
    # dct_y = np.zeros_like(y)
    block_size = 16
    dct_y = dct_function(y, block_size)

    # Stretch Cb and Cr arrays to match the size of the Y array
    cb_shrinked_shape = cb_shrinked.shape
    cr_shrinked_shape = cr_shrinked.shape
    directory = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)

    shape = dct_y.shape
    dct_y = dct_y.flatten()
    rle_dct = rle(dct_y)
    u_rle_dct_y = undo_rle(rle_dct).reshape(shape)

    cb_shape = cb_shrinked.shape
    cr_shape = cr_shrinked.shape
    dct_cb = dct_function(cb_shrinked, block_size)
    dct_cr = dct_function(cr_shrinked, block_size)
    np.savez_compressed(
        os.path.join(directory, file_name + ".custom2"),
        cb_shrinked_rle=rle(dct_cb.flatten()),
        cr_shrinked_rle=rle(dct_cr.flatten()),
        rle_dct=rle_dct,
        dct_y_shape=shape,
        cb_shape=cb_shape,
        cr_shape=cr_shape,
    )

    cb_stretched = stretch_array(
        undo_dct_function(dct_cb, block_size, cb_shape).reshape(cb_shrinked_shape), shape
    )
    cr_stretched = stretch_array(
        undo_dct_function(dct_cr, block_size, cr_shape).reshape(cr_shrinked_shape), shape
    )
    # reverse dct_y
    idct_y = undo_dct_function(u_rle_dct_y, block_size, shape)

    y = (idct_y + 128).clip(0, 255).astype(np.uint8)
    cb = (cb_stretched + 128).clip(0, 255).astype(np.uint8)
    cr = (cr_stretched + 128).clip(0, 255).astype(np.uint8)

    compressed_image = Image.merge(
        "YCbCr", (Image.fromarray(y), Image.fromarray(cb), Image.fromarray(cr))
    )
    compressed_image = compressed_image.convert("RGB")

    plt.imshow(compressed_image)
    plt.show()
    # compressed_image.show()
    compressed_image.save(os.path.join(directory, "compressed" + file_name), "JPEG")

    # return compressed_image


def pca_compress(image_path, quality=0):
    image = Image.open(image_path).convert("RGB")

    # Convert image to numpy array
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.show()

    r, g, b = split_image(image_array)
    channels = [0, 0, 0]
    channels = [tuple(), tuple(), tuple()]
    directory = os.path.dirname(image_path)
    file_name = os.path.basename(image_path).split(".")[0] + ".compressedim"

    for counter, array in enumerate((r, g, b)):
        channels[counter] = pca(array, quality)

    combined = combine_channels(*channels)
    image = Image.fromarray(combined)
    image.save(os.path.join(directory, "compressed" + file_name), "JPEG")


def svd_compress(image_path, quality=0):
    image = Image.open(image_path).convert("RGB")

    # Convert image to numpy array
    image_array = np.array(image)
    plt.imshow(image_array)
    plt.show()

    r, g, b = split_image(image_array)
    channels = [tuple(), tuple(), tuple()]
    for counter, array in enumerate((r, g, b)):
        channels[counter] = svd(array, quality)

    combined = combine_channels(*channels)
    image = Image.fromarray(combined)
    directory = os.path.dirname(image_path)
    file_name = os.path.basename(image_path)
    image.save(os.path.join(directory, "compressed" + file_name), "JPEG")
