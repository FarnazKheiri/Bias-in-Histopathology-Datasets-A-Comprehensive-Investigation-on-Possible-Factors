import numpy as np
import cv2
def to_gray(image, th):
    # Generate random color injection values for each channel
    red_injection = np.random.uniform(0.75, th)
    green_injection = np.random.uniform(0.75, th)
    blue_injection = np.random.uniform(0.75, th)

    # Split the image into individual color channels
    b, g, r = cv2.split(image)

    # Apply random injection of color to each channel
    b = np.clip(b * blue_injection, 0, 255).astype(np.uint8)
    g = np.clip(g * green_injection, 0, 255).astype(np.uint8)
    r = np.clip(r * red_injection, 0, 255).astype(np.uint8)

    # Merge the modified color channels into an RGB image
    modified_image = cv2.merge([b, g, r])

    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
    # expand the dimention
    img_array = np.expand_dims(gray_image, -1)
    # repeat the dimention
    img_array_3_channel = img_array.repeat(3, axis=-1)

    return img_array_3_channel
