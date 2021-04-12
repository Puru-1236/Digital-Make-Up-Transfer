import cv2
import numpy as np
import dlib

#control_triangle
#createMask

def control_triangle(img1):

    points_and_indices = []
    detector= dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    img=cv2.imread(img1)
    win_name='image'
    cv2.imshow(win_name, img)
    cv2.waitKey(1000)

    gray_image=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    mask=np.zeros_like(gray_image)
    face_data=detector(gray_image)

    for each_face in face_data:

        x1=each_face.left()
        y1=each_face.top()
        x2=each_face.right()
        y2=each_face.bottom()

        landmarks_points=[]
        cv2.rectangle(img=img, pt1=(x1,y1), pt2=(x2,y2), color=(0,255,0), thickness=2)
        cv2.imshow(win_name,img)
        # cv2.imwrite('step1.png', img)
        cv2.waitKey(2000)
        landmarks = predictor(image=gray_image, box=each_face)

        #cover all points now
        for point in range(0,68):

            x = landmarks.part(point).x
            y = landmarks.part(point).y
            landmarks_points.append((x,y))

            points = np.array([x, y])
            index = point
            point_and_index = (points, index)
            points_and_indices.append(point_and_index)

            cv2.circle(img=img, center=(x, y), radius=2, color=(0, 255, 0), thickness=-1)

        # cv2.imwrite('step2.png', img)
        points=np.array(landmarks_points, np.int32)
        convexhull=cv2.convexHull(points)
        cv2.fillConvexPoly(img=mask,points=convexhull,color=255)
        con_image=cv2.bitwise_and(img, img, mask=mask)

        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles1 = subdiv.getTriangleList()
        triangles = np.array(triangles1, dtype=np.int32)

        for point in triangles:
            pt1 = [point[0], point[1]]
            pt2 = [point[2], point[3]]
            pt3 = [point[4], point[5]]

            triangle = np.array([[pt1, pt2, pt3]], np.int32)
            cv2.polylines(img, [triangle], True, (0, 0, 255),1)
    # cv2.imwrite('step3.png', img)
    cv2.imshow(win_name,img)
    cv2.waitKey(2000)
    cv2.imshow("Con image", con_image)
    # cv2.imwrite('step4.png', con_image)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()

    return (points_and_indices, face_data, landmarks, triangles1)

def createMask(rows, cols, points_and_indices):

    left_eye=[];    right_eye=[];    cavity = [];    lips = []; nose = []
    left_brow=[];    right_brow=[];    skin = [];    masks = []

    for point_and_index in points_and_indices:
        if(17<=point_and_index[1]<=21):
            left_brow.append(point_and_index[0])
        elif(22<=point_and_index[1]<=26):
            right_brow.append(point_and_index[0])
        elif(27<=point_and_index[1]<=35):
            nose.append(point_and_index[0])
        elif(48<=point_and_index[1]<=59):
            lips.append(point_and_index[0])
        elif(36<=point_and_index[1]<=41):
            left_eye.append(point_and_index[0])
        elif(42<=point_and_index[1]<=47):
            right_eye.append(point_and_index[0])
        elif(60<=point_and_index[1]<=67):
            cavity.append(point_and_index[0])
        else:
            skin.append(point_and_index[0])

    for i, region in enumerate([left_eye, right_eye, cavity, lips, skin, left_brow, right_brow, nose]):

        mask = np.zeros(shape=(rows, cols), dtype=np.uint8)
        x = np.array(region)

        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                
                mask[i][j] = 1

        masks.append(mask)

    masks[3] -= masks[2]
    for i in range(8):
        if i != 4:
            masks[4] -= masks[i]

    return masks