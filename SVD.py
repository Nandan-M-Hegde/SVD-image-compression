import os
import sys
import humanize
from PIL import Image
import numpy as np
import decimal
import scipy.misc

# Taking the path of the image as input to the program
image_path = input("Enter the path of the image: ")
image_save_path = image_path.strip().split('\\')
name = image_save_path[-1].strip().split('.')
name[-2] = name[-2] + "_Compressed"
name[-1] = "jpg"
f_name = '.'.join(name)
image_save_path[-1] = f_name
image_save_path = '/'.join(image_save_path)

# Checking whether the specified image file exists or not at the specified directory
if not os.path.exists(image_path):
    print("File does not exist at the specified path")
    sys.exit()

# Opening the specified image file
img = Image.open(image_path)

# Getting the resolution of the image
col, row = img.size
print("Original image resolution: ", col, "x", row)

# Getting the original size of the image in bytes before compression
img_size = (os.path.getsize(image_path))

# img_size is in bytes.
# To print the image size in a more convenient way i.e., in Kilo Bytes we use humanize library
print("Original image size: ", humanize.naturalsize(img_size, binary=True), " (", img_size, " bytes)")

# Converting img to a matrix
im = np.array(img)

# Ask for confirmation
choice = input("Do you want to continue?[Y(es)]: ")
if (choice != 'y') and (choice != 'Y'):
    print("Exiting")
    sys.exit()

# Calculate the best rank 'k'
k = int(((row * col) / (row + col + 1)))

# Ask for the value of k
print("Enter the rank (k) of the compressed image")
print("1 to ", min(row, col), " (0 will automatically select the best rank 'k')")
k1 = int(input())
if 0 < k1 <= min(row, col):
    k = k1
print("k = ", k)

print("Compressing.......")

try:
    # Since RGB values range from 0 to 255 we divide it by 255 to bring the values in the range [0,1]
    im = im / 255

    # Separating red, green and blue color values as 3 separate matrices
    img_red = im[:, :, 0]
    img_green = im[:, :, 1]
    img_blue = im[:, :, 2]

    # Now we have 3 matrices having corresponding RGB values
    # Now we perform SVD on these 3 matrices separately
    U_r, Sigma_r, V_r = np.linalg.svd(img_red, full_matrices=True)
    U_g, Sigma_g, V_g = np.linalg.svd(img_green, full_matrices=True)
    U_b, Sigma_b, V_b = np.linalg.svd(img_blue, full_matrices=True)

    # Selecting k columns from the matrices obtained in the previous step
    U_r_k = U_r[:, 0:k]
    U_g_k = U_g[:, 0:k]
    U_b_k = U_b[:, 0:k]

    # Selecting k rows and columns from the matrices Sigma_r, Sigma_g, Sigma_b
    Sigma_r_k = Sigma_r[0:k]
    Sigma_g_k = Sigma_g[0:k]
    Sigma_b_k = Sigma_b[0:k]

    # Selecting k rows from the matrices V_r, V_g, V_b
    V_r_k = V_r[0:k, :]
    V_g_k = V_g[0:k, :]
    V_b_k = V_b[0:k, :]

    # Reconstructing the matrix corresponding to red color by combining U_r_k, Sigma_r_k, V_r_k
    img_red_k = np.dot(U_r_k, np.dot(np.diag(Sigma_r_k), V_r_k))

    # Reconstructing the matrix corresponding to green color by combining U_g_k, Sigma_g_k, V_g_k
    img_green_k = np.dot(U_g_k, np.dot(np.diag(Sigma_g_k), V_g_k))

    # Reconstructing the matrix corresponding to blue color by combining U_b_k, Sigma_b_k, V_b_k
    img_blue_k = np.dot(U_b_k, np.dot(np.diag(Sigma_b_k), V_b_k))

    # Reconstructing the RGB matrix
    # Creating a container matrix with all entries as zeroes
    img_recon = np.zeros((row, col, 3))

    # Inserting the RGB values into the container matrix
    img_recon[:, :, 0] = img_red_k
    img_recon[:, :, 1] = img_green_k
    img_recon[:, :, 2] = img_blue_k

    # Getting the size of the matrix (i.e row, column) tells us the resolution of the image
    row, col, _ = img_recon.shape

    # Converting the matrix to image and saving the image at image_save_path
    scipy.misc.toimage(img_recon).save(image_save_path)

except Exception as e:
    print(e)
    print("Compression failed!")
    sys.exit()

if not os.path.exists(image_save_path):
    print("Could not save image")
    sys.exit()

print("Done!")
print("File saved to ", image_save_path)

# Print the resolution of compressed image
print("Compressed image resolution: ", col, "x", row)

# Getting the size of compressed image
c_img_size = os.path.getsize(image_save_path)
print("Compressed image size: ", humanize.naturalsize(c_img_size, binary=True), " (", c_img_size, " bytes)")

# Calculate the compression ratio
CR = decimal.Decimal(img_size / c_img_size)
print("Compression ratio is ", round(CR, 2))
