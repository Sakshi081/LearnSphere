import cv2
import os

# Paths for input images and output directory
image1_path = 'static/std1a.png'
image2_path = 'static/std1b.png'
output_dir = 'solution'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the two images
image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)

# Check if images are loaded
if image1 is None or image2 is None:
    print("Error: One or both images could not be loaded.")
    exit()

# Resize the second image to match the first
image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# Convert both images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

# Compute absolute difference
diff = cv2.absdiff(gray1, gray2)

# Threshold the difference to highlight regions
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Find contours of the differences
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the detected differences on both images
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    center_x = x + w // 2
    center_y = y + h // 2
    radius = max(w, h) // 2  # Use the larger of width or height as radius
    # Draw circles around the differences
    cv2.circle(image1, (center_x, center_y), radius, (0, 255, 0), 2)
    cv2.circle(image2_resized, (center_x, center_y), radius, (0, 255, 0), 2)

# Save the marked images in the solutions directory
output_image1_path = os.path.join(output_dir, 'std1a_marked.png')
output_image2_path = os.path.join(output_dir, 'std1b_marked.png')

cv2.imwrite(output_image1_path, image1)
cv2.imwrite(output_image2_path, image2_resized)

print(f"Marked images saved to: {output_dir}")
