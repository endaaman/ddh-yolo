import invoke
from glob import glob

from endaaman.utils import prod_fmt

@invoke.task
def train_all(c):
    for depth in ['l', 'm', 's']:
        CFG = f'yolov5/models/yolov5{depth}.yaml',
        for i in range(6):
            num = i + 1
            if depth == 'l' and i<3:
                continue
            CFG = f'yolov5/models/yolov5{depth}.yaml'
            YAML = f'fold{num}.yaml'
            NAME = f'{depth}_fold{num}'
            EPOCH = 300
            BATCH_SIZE = 32
            cmd = f'python yolov5/train.py --data {YAML} --cfg {CFG}' \
                f' --batch-size {BATCH_SIZE} --epochs {EPOCH} --name {NAME}'
            print(cmd)
            invoke.run(cmd)


@invoke.task
def detect_all(c):
    for i in range(6):
        NUM = i + 1
        SIZE = 'l'
        NAME= f'{SIZE}_fold{NUM}'
        W = f'yolov5/runs/train/{SIZE}_fold{NUM}/weights/best.pt'
        cmd = f'python yolov5/detect.py --source "data/folds6/fold{NUM}/test/images/*.jpg"' \
            f' --weight {W} --hide-label --name {NAME}' \
            f' --save-txt --exist-ok --save-conf --conf 0.3'
        print(f'CMD: {cmd}')
        invoke.run(cmd)
