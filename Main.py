import cv2
import sys
import numpy as np
from numpy import float32
from lib_functions import *
from boundary import *

image1 = sys.argv[1]
image2 = sys.argv[2]
image_b = cv2.imread(image1)
image_a = cv2.imread(image2)
shape = image_b.shape
a=[0, 0, -1, 0, 0]
b= [0, -1, -2, -1, 0]
c= [-1, -2, 16, -2, -1]

kernel = np.array([a, b, c,b,a])
index_img_b = control_triangle(image1)
index_img_a = control_triangle(image2)


img_a_warped = merging(image_a, image_b, index_img_a, index_img_b )
cv2.imshow('warped_image', img_a_warped/255)



layers_img_b = bilateral_Filter(image_b)
layers_img_a = bilateral_Filter(img_a_warped.astype('uint8'))

sub_delta_i = 0
exa_delta_e = 1
zeros = np.zeros(img_a_warped.shape, dtype=img_a_warped.dtype)

skin_detail = np.where(True, sub_delta_i*layers_img_b[1] + exa_delta_e*layers_img_a[1], zeros[:,:,0])
skin_detail = skin_detail.astype(layers_img_b[1].dtype)

masks = createMask(shape[0], shape[1], index_img_b[0])
color = color_transfer(masks, layers_img_b[2], layers_img_a[2], layers_img_a[2].shape)


highlight_shading = highlight_shading_transfer(masks[4], layers_img_b[0], layers_img_b[0].shape,
                                                                cv2.filter2D(layers_img_a[0],-1,kernel), cv2.filter2D(layers_img_b[0],-1,kernel))

lips_transfer = np.multiply((layers_img_b[1] + layers_img_a[0]), masks[3])


temp = np.ones(shape, dtype='uint8')
temp = temp[:, :, 0]-masks[3]
highlight_shading = np.multiply(highlight_shading, temp)
composed = (highlight_shading + lips_transfer + skin_detail).astype('uint8')

shape = (composed.shape[0], composed.shape[1], 3)
output = np.ndarray(shape=shape, dtype='uint8')
output[:, :, 0] = composed
output[:, :, 1] = color[:, :, 0]
output[:, :, 2] = color[:, :, 1]

output = cv2.cvtColor(output, cv2.COLOR_LAB2RGB)
output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

cv2.imshow('result image', output)
cv2.imwrite('result_image.png', output)

cv2.waitKey(0)
cv2.destroyAllWindows()