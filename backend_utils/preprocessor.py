import cv2
from backend_utils.utils import img_map


def resize(src_imgs, width, height):
    return img_map(src_imgs, cv2.resize, dsize=(width, height))


def cvt_2_gray(src_imgs):
    try:
        return img_map(src_imgs, cv2.cvtColor, code=cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return src_imgs


def bilateral(src_imgs, d=9, sigmaColor=75, sigmaSpace=75):
    return img_map(src_imgs, cv2.bilateralFilter, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)


def gaussian_blur(src_imgs, ksize=(3, 3)):
    return img_map(src_imgs, cv2.GaussianBlur, ksize=ksize, sigmaX=0)


def laplacian(src_imgs, ksize=5):
    return img_map(src_imgs, cv2.Laplacian, ddepth=cv2.CV_8U, ksize=ksize)


def value_equalize(src_imgs):
    def v_e(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        v = cv2.equalizeHist(v)
        img = cv2.merge([h, s, v])
        return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

    return img_map(src_imgs, v_e)
