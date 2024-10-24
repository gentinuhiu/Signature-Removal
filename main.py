import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure, morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
from pdf2image import convert_from_path

# Example usage
pdf_path = 'input.pdf'
jpg_path = 'image.jpg'
images = convert_from_path(pdf_path)
image = images[0]
image.save(jpg_path, 'JPEG')

# Read Image
img = cv2.imread(jpg_path, 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

# Extract Blobs
blobs = img > img.mean()
blobs_labels = measure.label(blobs, background=1)
image_label_overlay = label2rgb(blobs_labels, image=img)

# Area of Text
the_biggest_component = 0
total_area = 0
counter = 0
average = 0.0
for region in regionprops(blobs_labels):
    if region.area > 10:
        total_area += region.area
        counter += 1
    if region.area >= 250:
        if region.area > the_biggest_component:
            the_biggest_component = region.area

# Threshold
if counter > 0:
    average = total_area / counter
print("the_biggest_component: " + str(the_biggest_component))
print("average: " + str(average))

a4_constant = ((average / 84.0) * 250.0) + 100
print("a4_constant: " + str(a4_constant))

# Remove Noise
b = morphology.remove_small_objects(blobs_labels, a4_constant)

# Save the processed image
plt.imsave('pre_version.png', b)

# Read the pre-processed image
img2 = cv2.imread('pre_version.png', 0)
# Ensure binary
img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# Save the result
cv2.imwrite("signature_detection.png", img2)

# Compute and save difference
diff = cv2.bitwise_xor(img, img2)
diff = cv2.bitwise_not(diff)
cv2.imwrite("signature_removal.png", diff)

# Function to display image with matplotlib
def display_image(title, img, resize_factor=0.5):
    # Compute new size
    h, w = img.shape[:2]
    new_w = int(w * resize_factor)
    new_h = int(h * resize_factor)
    
    # Resize image for display
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Display image
    plt.figure(figsize=(new_w / 100, new_h / 100))  # Adjust figsize to fit the resized image
    plt.title(title)
    plt.imshow(img_resized, cmap='gray')
    plt.axis('off')  # Hide axis
    plt.show()

# Display images using matplotlib
display_image("Original Image", img)
display_image("Signature Detection", img2)
display_image("Signature Removal", diff)
