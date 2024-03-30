import pybboxes as pbx
# pascal_voc: four values in pixels: [x_min, y_min, x_max, y_max]. 
voc_bbox = (100, 100, 200, 200)
W, H = 1000, 1000  # WxH of the image
pbx.convert_bbox(voc_bbox, from_type="voc", to_type="yolo", image_size=(W,H))