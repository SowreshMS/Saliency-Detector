import cv2
import matplotlib.pyplot as plt

# Load an image
image_path = ""  # Replace with the actual path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Normalize the image to [0, 1]
image = image / 255.0

# Convert the image to 8-bit unsigned integer
image = cv2.convertScaleAbs(image * 255)

# Initialize the Spectral Residual Saliency Detector
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

# Compute the saliency map
(success, saliencyMap) = saliency.computeSaliency(image)

# Rescale the saliency map to the range [0, 1]
saliencyMap = saliencyMap / saliencyMap.max()

# Display the original image and the saliency map
plt.subplot(121), plt.imshow(image), plt.title('Original Image')
plt.subplot(122), plt.imshow(saliencyMap, cmap='gray'), plt.title('Saliency Map')

# set new widths and heights to downscale the image
w = 200
h = 100
saliencyMap = cv2.resize(saliencyMap, (w, h))

# Write saliency map values to a text file
output_file_path = ""  # Replace with the desired output file path

with open(output_file_path, 'w') as file:
    height, width = saliencyMap.shape
    for y in range(height):
        for x in range(width):
            value = saliencyMap[y, x]
            file.write(f"{value},")
        file.write('\n')

print(f"Saliency values have been written to: {output_file_path}")

plt.show()
