from Stitcher import Stitcher
import cv2

# 读取图片
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

# 图片拼接
stitcher = Stitcher()
result, vis = stitcher.stitch([img1, img2], showMatches=True)

# cv2.imshow('img1', img1)
# cv2.imshow('img2', img2)
cv2.imwrite('keypoints matches.jpg', vis)
cv2.imwrite('result.jpg', result)
# cv2.waitKey(0)
# cv2.destroyWindow()
print('finish')

