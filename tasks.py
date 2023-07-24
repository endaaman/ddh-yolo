import invoke

@invoke.task
def train_all(c):
    for i in range(0, 5):
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
