from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import scipy as sp

TEST_IMG = cv2.imread("/home/jmend/code/makemit/test.png")

class Tracker(object):
    def __init__(self, original_image):
        self._orignal_image = original_image
        self._gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    def detect_features(self, sub_image):
        sub_image_uint8 = np.uint8(sub_image)
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(sub_image_uint8, None)
        kp2, des2 = orb.detectAndCompute(self._gray_image, None)

        return (kp1, des1, kp2, des2)

    def match(self, des1, des2, k=2):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.85*n.distance:
                good.append([m])
        return good

    def conv_to_pts(self, kp1, kp2, goodMatches):
        src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in goodMatches ]).reshape(-1,1,2) 
        return src_pts, dst_pts

    def getAffineTransform(self, src_pts, dst_pts):
        trans = cv2.estimateRigidTransform(src_pts, dst_pts, False)
        return trans

    def applyAffineTransform(self, image, transform):
        image = np.float32(image)
        timage = cv2.warpAffine(image, transform, image.shape)
        return timage

    def transform(self, image):
        kp1, des1, kp2, des2 = self.detect_features(image)
        goodMatches = self.match(des1, des2)
        img3 = cv2.drawMatches(image, kp1, self._gray_image, kp2, goodMatches, flags=2)
        src, dst = self.conv_to_pts(kp1, kp2, goodMatches)
        affineTransform = self.getAffineTransform(src, dst)
        x = -affineTransform[0][2]
        y = -affineTransform[1][2]
        theta = np.arccos(affineTransform[0,0])
        return x, y, theta

    def xyt_to_affine(self, x, y, theta):
        mat = np.array([
            [np.cos(theta), np.sin(theta), -x],
            [-np.sin(theta), np.cos(theta), -y]
        ])
        return mat

if __name__ == "__main__":
    t = Tracker(TEST_IMG)
    slc = np.concatenate((np.ones([600, 300]) * 255, t._gray_image[:600, :300]), axis=1)
    # slc = np.copy(t._gray_image[:300, :150])
    # slc = slc + np.random.randint(-6, 6, slc.shape)
    # slc[slc > 255] = 255
    # slc[slc < 0] = 0
    # slc = np.uint8(slc)


    plt.imshow(slc, cmap="Greys_r")

    M = cv2.getRotationMatrix2D((300, 300), -20.0, 1.0)
    slc = cv2.warpAffine(slc, M,(slc.shape[1], slc.shape[0]), borderValue=255)
    slc = slc.astype(np.uint8)

    plt.imshow(slc, cmap="Greys_r")
    plt.show()

    trans = t.transform(slc)
    print("trans", trans)
