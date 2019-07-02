import os
import cv2
import numpy as np
import pandas as pd

path1 = "D:\\Sanjeev\\Saliency Object Detection\\SaliencyMap\\binary"
path = "D:\\Sanjeev\\Saliency Object Detection\\SaliencyMap\\test"

final = []

for image_path in os.listdir(path):
        input_path = os.path.join(path, image_path)
        im1 = cv2.imread(input_path,0)                    # Ground truth image
        path2 = os.path.join(path1, 'bfcrop'+image_path[:-4] + '.jpg')
        im2 = cv2.imread(path2,0)                         # Obtained image
        print(image_path)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        r,c = im1.shape
        
        for i in range(r):
            for j in range(c):                
                if im1[i,j]==im2[i,j]==255:
                    tp = tp + 1
                elif im1[i,j]==im2[i,j]==0:
                    tn = tn + 1
                elif im1[i,j]!=im2[i,j] and im2[i,j]==255:
                    fp = fp + 1
                else:
                    fn = fn + 1
                    
        sensitivity = round(tp/(tp + fn),2)
        specificity = round(tn/(tn + fp),2)
        accuracy = round((tp + tn)/(tp + tn +fp +fn),2)
        features = np.array([image_path,sensitivity,specificity,accuracy])
        final.append(features)


df = pd.DataFrame(final)

filepath = "final.xlsx"  # path where to save the features
df.to_excel(filepath)  # write the extracted features+labels as excel file        
        