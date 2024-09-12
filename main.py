from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np

background = cv2.imread('test/ps2_case.jpg')
image_to_overlay = cv2.imread('covers/aoe2.jpg')


height, width = image_to_overlay.shape[:2]
# Define the source points (corners of the image_to_overlay)
src_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')


# top-left  823x 650y
# top-right 1915x 650y
# bot-left  823x 2189y
# bot-right 1915x 2189y
# Define the destination points (area in the background image where you want the image_to_overlay to fit)
dst_pts = np.array([[823, 645], [1918, 642], [1918, 2192], [823, 2198]], dtype='float32')

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Perform the perspective warp to transform the image_to_overlay
warped_image = cv2.warpPerspective(image_to_overlay, M, (background.shape[1], background.shape[0]))

# Create a mask of the warped image (assuming no transparency in the image_to_overlay)
# You can modify this part if you need to handle alpha channels or transparency
mask = np.zeros((background.shape[0], background.shape[1]), dtype=np.uint8)
cv2.fillConvexPoly(mask, dst_pts.astype(int), 255)

# Invert the mask to get the region of the background to keep
inverse_mask = cv2.bitwise_not(mask)

# Use the mask to extract the region of interest (ROI) from the background
background_region = cv2.bitwise_and(background, background, mask=inverse_mask)

# Use the mask to extract the warped image
warped_region = cv2.bitwise_and(warped_image, warped_image, mask=mask)

# Add the background region and the warped image together
final_image = cv2.add(background_region, warped_region)

# Display the result

cv2.imwrite('final_game_case.jpeg', final_image)

# cv2.imshow('Overlay Result', final_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


