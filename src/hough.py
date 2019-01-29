import cv2
import numpy as np
import matplotlib.pyplot as plt
from util import *

# load image, grayscale and crop unnecessary white regions

cmap='gray'
for i in range(5):
	img = cv2.imread(f'data/I_test_{i+1}.png')
	img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img_cropped = crop_borders(img)
	plt.subplot(1,5,i+1)
	plt.imshow(img_cropped, cmap)
	plt.axis('off')
	plt.title(f'Test {i+1}')

plt.get_current_fig_manager().window.showMaximized()
plt.show()

# import cv2
# import numpy as np
#
# img = cv2.imread('data/I_test_5.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
#
# lines = cv2.HoughLines(edges,1,np.pi/180,200)
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
