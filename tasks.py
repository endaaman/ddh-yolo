import invoke
from glob import glob

@invoke.task
def train_all(c):
    for i in range(6):
        num = i + 1
        CFG = 'yolov5/models/yolov5s.yaml'
        YAML = f'fold{num}.yaml'
        NAME = f'fold{num}'
        EPOCH = 200
        BATCH_SIZE = 32

        cmd = f'python yolov5/train.py --data {YAML} --cfg {CFG}' \
            f' --batch-size {BATCH_SIZE} --epochs {EPOCH} --name {NAME}'

        invoke.run(cmd)
        # print(cmd)


@invoke.task
def detect_all(c):
    for i in range(6):
        NUM = i + 1
        cmd = f'python yolov5/detect.py --source "data/folds6/fold{NUM}/test/images/*.jpg"' \
            f' --weight yolov5/runs/train/fold{NUM}/weights/best.pt --hide-label --name fold{NUM}' \
            f' --save-txt --exist-ok'
        print(f'CMD: {cmd}')
        invoke.run(cmd)
