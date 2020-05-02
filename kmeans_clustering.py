import numpy as np
import cv2
import matplotlib.pyplot as plt
# from drawRect import drawRect

def KmeansSegment(image):
    original_image = cv2.imread(image)
    print("original image: ", original_image.shape)

    img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)

    vectorized = img.reshape((-1,3))

    vectorized = np.float32(vectorized)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    K = 2
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)
    label = label.reshape((img.shape[0],img.shape[1]))
    # print("label shape: ", label.shape)
    # print("label unique: ", np.unique(label))
    # print("center: ", center)

    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    # print("res2 pixel values: ", res2[0,0])

    ind = np.argwhere(label == 0)
    xmin = np.min(ind[:,0])
    xmax = np.max(ind[:,0])
    ymin = np.min(ind[:,1])
    ymax = np.max(ind[:,1])
    # print(xmin,ymin,xmax,ymax)
    res2 = cv2.rectangle(res2, (ymin,xmin),(ymax,xmax),(255,0,0),1)

    figure_size = 10
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(res2)
    plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    image = "400_pixels.png"
    KmeansSegment(image)