import numpy as np
import cv2

image = cv2.imread('测试头像图片路径') 
#原图行数列数
rows = image.shape[0]
cols = image.shape[1] 
#新图平铺2行三列，即新图行数变为2倍，列数变为3倍
new_rows = rows * 2
new_cols = cols * 3 

#生成新图的数组
new_image = np.zeros(shape=(new_rows, new_cols, 3), dtype=np.uint8) 

#复制原图的每一个像素
row = 0
col = 0 

for now_row in range(new_rows): 
    for now_col in range(new_cols):
        new_image[now_row, now_col, 0] = image[row, col, 0]
        new_image[now_row, now_col, 1] = image[row, col, 1]
        new_image[now_row, now_col, 2] = image[row, col, 2]
        col+=1 
        #超过原图列数范围，归0，重新开始复制 
        if col>=cols:
            col=0

    row+=1 
    #超过原图行数范围，归0，重新开始复制 
    if row>=rows:
        row=0

cv2.imshow('new image', new_image)
cv2.waitKey()
#cv2.destroyAllWindows()