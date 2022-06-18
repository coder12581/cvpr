import  numpy as np
import cv2

class Stitcher:

    # 拼接函数
    def stitch(self, images, ratio = 0.75, reprojThresh = 4.0, showMatches = False):
        # 读取图像
        imageB, imageA = images
        # 计算特征点和特征向量
        kpsA, featureA = self.detectAndDescribe(imageA)
        kpsB, featureB = self.detectAndDescribe(imageB)

        # 匹配两张图片的特征点
        M = self.matchKeypoints(kpsA, kpsB, featureA, featureB, ratio, reprojThresh)

        # 没有匹配点，退出
        if not M:
            return None

        matches, H, status = M
        # 将图片A进行视角变换 中间结果
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # 将图片B传入]
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        self.cv_show('result', result)

        # 检测是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return result, vis

        # 返回匹配结果
        return result


    def detectAndDescribe(self, image):
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.SIFT_create()
        # 检测特征点并计算描述子
        kps, features = descriptor.detectAndCompute(gray, None)

        kps = np.float32([kp.pt for kp in kps])

        return kps, features

    def matchKeypoints(self, kpsA, kpsB, featureA, featureB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.BFMatcher()

        # 使用KNN检测来自AB图的SIFT特征匹配
        rawMatches = matcher.knnMatch(featureA, featureB, 2)

        # 过滤
        matches = []
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算H矩阵
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            return matches, H, status

    # 展示图像
    def cv_show(self,name, img):
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片，将A、B图左右连接到一起
        hA, wA = imageA.shape[:2]
        hB, wB = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历，画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上
            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis