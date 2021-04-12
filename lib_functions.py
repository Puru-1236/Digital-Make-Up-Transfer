import numpy as np
import cv2
from numpy import float32

#bilateral_Filter
#color_transfer
#highlight_shading_transfer
#warping
#merging

def bilateral_Filter(inputImage):
    
    cielab_img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2Lab)
    structure_layer=cv2.bilateralFilter(src=cielab_img[:,:,0],d=15,sigmaColor =80,sigmaSpace= 80)
    detail_layer = cielab_img[:,:,0] - structure_layer

    return [structure_layer, detail_layer, cielab_img[:, :, 1:3]]

def color_transfer(masks, clrSub, clrExp, shape, r = 0.8):

    clrRes = np.zeros(shape=shape, dtype='uint8')
    mask = masks[0]

    for i in range(1,3):
        mask += masks[i]

    for i in range(shape[0]):
        for j in range(shape[1]):
            clrRes[i, j] = (1-r)*clrSub[i, j] + r*clrExp[i, j] if(mask[i, j] == 0) else clrSub[i, j]

    return clrRes


def dog(img, size=(5,5), k=1.6, sigma=0.5, gamma=0):
	img1 = cv2.GaussianBlur(img,size,0)
	img2 = cv2.GaussianBlur(img,size,sigma*k)
	return (img1)

def xdog(img, sigma=0.5, k=1.6, gamma=1, epsilon=1, phi=1):
	aux = dog(img, sigma = sigma, k = k, gamma = gamma);aux1 = dog(img)
	for i in range(aux.shape[0]):
		for j in range(aux.shape[1]):
			if(aux[i,j] < epsilon):
				aux[i,j] = 1*255
			else:
				aux[i,j] = 255*(1 + np.tanh(phi*(aux[i,j])))
	return aux1

def highlight_shading_transfer(beta, structureSubject, shape, structExample, structSubject):

    structResult = np.ndarray(shape=shape, dtype='uint8')
    gauss_structureSubject = xdog(structureSubject,sigma=0.4,k=1.6, gamma=0.5,epsilon=-0.5,phi=10)

    for i in range(shape[0]):
        for j in range(shape[1]):           
            st_exp = abs(structExample[i, j]);   st_sub = abs(structSubject[i, j])
            structResult[i, j] = (structExample[i, j])/2 if(st_exp*beta[i, j] > st_sub) else (structSubject[i, j])/2

    return structResult + gauss_structureSubject

def warping(example, subject, exp, sub):

    warped_tri = sub[3]
    input_tri = exp[3]
    output = np.ones(example.shape, dtype=float32)

    for i in range(warped_tri.shape[0]):

        dst = np.reshape(a= warped_tri[i, :], newshape=(3, 2))
        tri2 = dst.astype(float32)

        pt_num = [k[1] for pt in tri2 for k in sub[0] if(pt[0] == k[0][0] and pt[1] == k[0][1]) ]

        src = np.ndarray(shape=(3, 2))
        for i, num in enumerate(pt_num):
            src[i, :] = exp[0][num][0]
        tri1 = src.astype(dtype=float32)

        r1 = cv2.boundingRect(array=tri1)
        (x1, y1, w1, h1) = r1
        r2 = cv2.boundingRect(array=tri2)
        (x2, y2, w2, h2) = r2
        tri1Cropped = []
        tri2Cropped = []

        tri1Cropped.append(((tri1[0][0] - x1),(tri1[0][1] - y1)))
        tri1Cropped.append(((tri1[1][0] - x1),(tri1[1][1] - y1)))
        tri1Cropped.append(((tri1[2][0] - x1),(tri1[2][1] - y1)))

        tri2Cropped.append(((tri2[0][0] - x2),(tri2[0][1] - y2)))
        tri2Cropped.append(((tri2[1][0] - x2),(tri2[1][1] - y2)))
        tri2Cropped.append(((tri2[2][0] - x2),(tri2[2][1] - y2)))
        

        img1Cropped = example[y1:y1 + h1, x1:x1 + w1]
        # cv2.imshow('cropped',img1Cropped)
    
        warpMat = cv2.getAffineTransform(np.float32(tri1Cropped), np.float32(tri2Cropped))
        img2Cropped = cv2.warpAffine( src= img1Cropped,M= warpMat,dsize= (w2, h2),dst= None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101, borderValue=None )
        mask = np.zeros(shape= (h2, w2, 3), dtype = np.float32)

        cv2.fillConvexPoly(img= mask,points= np.int32(tri2Cropped),color= (1.0, 1.0, 1.0),lineType= 16,shift= 0)
       
        output[y2:y2+h2, x2:x2 + w2] =  output[y2:y2 + h2, x2:x2+w2] * ( 1 - mask ) + img2Cropped * mask

    return output

def merging(example, subject, ctr_e, ctr_s):

    subject_face = warping(subject, subject, ctr_s, ctr_s)
    output = subject - subject_face
    warped_example = warping(example, subject, ctr_e, ctr_s)
    output = output + warped_example

    return output