# !/bin/bash
echo 'yolov5s regression......'
echo 'int8 bmodel'
python3 det_yolov5.py --bmodel $1 --imgdir /home/database/coco/images/val2017/ --tpu_id 0 --input /home/database/annotations/instances_val2017.json --result $2
echo 'calulate mAP'
python3 calc_map.py --anno /home/database/annotations/instances_val2017.json --log $2 --image-dir /home/database/coco/images/val2017/
