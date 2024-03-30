import cv2
import os

input_dir = "/home/openark/Desktop/yolo_bbox/test_original"
output_dir = "/home/openark/Desktop/yolo_bbox/test"
for file in os.listdir(input_dir):
    img = cv2.imread(os.path.join(input_dir, file))
    im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("d", im_bgr)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(output_dir, file.split(".")[0] + ".png"), im_bgr)

