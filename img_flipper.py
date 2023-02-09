import cv2 as cv
import glob
import re
import os

# TO DO: Update paths as required

path = "/home/morgan/rvss2023/rvss/RVSS_Need4Speed/on_laptop/new_data/*.jpg"
save_path = "/home/morgan/rvss2023/rvss/RVSS_Need4Speed/on_laptop/augmented_data/"
img_count = 0

for img_path in glob.glob(path):
    print(img_path)

    img = cv.imread(img_path)
    f = os.path.basename(img_path)
    steering = re.findall("-\d.\d\d", f)

    if steering == []:
        steering = re.findall("\d\D\d\d", f)
    steering = float(steering[0])

    if steering < 0 or steering > 0:
        cv.imwrite(save_path + str(img_count) + str(steering) + '.jpg' , img)
        img_count = img_count + 1
        steering = steering * -1
        cv.imwrite(save_path + str(img_count) + str(steering) + '.jpg' , cv.flip(img, 1) )
        img_count = img_count + 1
    else:
        cv.imwrite(save_path + str(img_count) + str(steering) + '.jpg' , img )
        img_count = img_count + 1
