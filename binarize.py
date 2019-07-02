import cv2
import os

path = "D:\\Sanjeev\\Saliency Object Detection\\SaliencyMap\\b"
outPath = "D:\\Sanjeev\\Saliency Object Detection\\SaliencyMap\\binary"

for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path)
        img = cv2.imread(input_path,0)
        ret,thresh = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        fullpath = os.path.join(outPath, 'b'+image_path)
        cv2.imwrite(fullpath,thresh)
        
        