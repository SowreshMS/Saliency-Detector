import cv2
import matplotlib.pyplot as plt

# Load an image
image_path = "" # Replace with the actual path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the Spectral Residual Saliency Detector
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

# Compute the saliency map
(success, saliencyMap) = saliency.computeSaliency(image)

# Rescale the saliency map to the range [0, 255]
saliencyMap = (saliencyMap * 255).astype("uint8")

# Display the original image and the saliency map
plt.imshow(image)
plt.show()
plt.imshow(saliencyMap)

# Set new widths and heights to downscale the image
w = 200
h = 100
saliencyMap = cv2.resize(saliencyMap, (w, h))

# Write (r, g, b) pixel values to a text file
output_file_path = ""  # Replace with the desired output file path

with open(output_file_path, 'w') as file:
    height, width = saliencyMap.shape
    for y in range(height):
        for x in range(width):
            value = saliencyMap[y, x]
            file.write(f"{value},")
        file.write('\n')

print(f"Saliency values have been written to: {output_file_path}")
