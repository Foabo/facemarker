import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2

# data/test/test_160/CR7/65c1ef9ef6c144b4a07d6ff188c938fc.png 499 68 794 474
# data/test/test_160/CR7/p_fD-hnfikve2080614.png 267 5 364 124
l_x = 499
l_y = 68
r_x = 794
r_y=474
import cv2
image = cv2.imread('data/test/raw/CR7/65c1ef9ef6c144b4a07d6ff188c938fc.jpeg')
cv2.rectangle(image, (l_x, l_y), (r_x, r_y), (255, 0, 0), 3)
cv2.imwrite('data/test/raw/CR7/1.jpg', image)
