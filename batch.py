import os
import shutil
from glob import glob
from itertools import combinations

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.model_selection import StratifiedKFold
from pydantic import Field
from matplotlib import pyplot as plt
from tqdm import tqdm
from mean_average_precision import MetricBuilder

from endaaman.ml import BaseMLCLI


J = os.path.join

def read_bbs(filename):
    with open(filename, 'r') as f:
        output = f.read()
    bbs = []
    for line in output.split('\n'):
        tokens = line.split(' ')
        if len(tokens) != 5:
            continue
        id = tokens[0]
        anchors = [float(v) for v in tokens[1:]]
        bbs.append({
            'id': id,
            'anchors': anchors,
        })
    return bbs

def write_bbs(bbs, filename):
    lines = []
    for bb in bbs:
        tokens = []
        tokens.append(str(bb['id']))
        anchors = [f'{a:.6f}' for a in bb['anchors']]
        tokens += anchors
        lines.append(' '.join(tokens))

    text = '\n'.join(lines)
    with open(filename, 'w') as f:
        f.write(text)


def read_label_as_df(path, with_confidence=False):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cols = ['label', 'x0', 'y0', 'x1', 'y1']
    if with_confidence:
        cols += ['conf']
    data = []
    for line in lines:
        parted  = line.split(' ')
        class_id = int(parted[0]) + 1
        center_x, center_y, w, h = [float(v) for v in parted[1:5]]
        data.append([
            # convert yolo to pascal voc
            class_id,
            center_x - w / 2,
            center_y - h / 2,
            center_x + w / 2,
            center_y + h / 2,
        ])
        if with_confidence:
            data[-1].append(float(parted[-1][:-1]))
    return pd.DataFrame(columns=cols, data=data)


def validate_bb(df):
    ok = True
    for label in range(1, 7):
        b = df[df['label'] == label]
        if len(b) == 1:
            continue
        if len(b) > 1:
            ok = False
        elif len(b) == 0:
            ok = False
    return ok



# def bbdf_to_str(df):
#     lines = []
#     for idx, row in df.iterrows():
#         x0, y0, x1, y1 = row[['x0', 'y0', 'x1', 'y1']]
#         cls_ = row['label']
#         x = (x1 - x0) / 2
#         y = (y1 - y0) / 2
#         w = x1 - x0
#         h = y1 - y0
#         line = f'{cls_} {x:.6f} {y:.6f} {w:.6f} {h:.6f}'
#         lines.append([cls_, line])
#     lines = sorted(lines, key=lambda v:v[0])
#     return '\n'.join([l[1] for l in lines])



def calc_iou(box1, box2):
    intersection_x1 = max(box1[0], box2[0])
    intersection_y1 = max(box1[1], box2[1])
    intersection_x2 = min(box1[2], box2[2])
    intersection_y2 = min(box1[3], box2[3])
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = (intersection_width * intersection_height) / (area_box1 + area_box2 - (intersection_width * intersection_height))
    return iou

class CLI(BaseMLCLI):
    class CommonArgs(BaseMLCLI.CommonArgs):
        pass

    def run_copy(self, a):
        df = pd.read_excel('data/table.xlsx', converters={'label': str})
        for i, row in df.iterrows():
            id = row['label']
            src_image = f'data/images/{label}.jpg'
            src_label = f'data/labels/{label}.txt'
            subdir = 'test' if row['test'] else 'train'
            os.makedirs(f'data/yolo/{subdir}/images', exist_ok=True)
            os.makedirs(f'data/yolo/{subdir}/labels', exist_ok=True)
            dest_image = f'data/yolo/{subdir}/images/{label}.jpg'
            dest_label = f'data/yolo/{subdir}/labels/{label}.txt'
            shutil.copyfile(src_image, dest_image)
            shutil.copyfile(src_label, dest_label)

    def run_crop512(self, a):
        for fold in range(1, 7):
            for t in ['train', 'test']:
                src = f'data/folds6/fold{fold}/{t}/images/'
                dest = f'data/folds6_512/fold{fold}/{t}/images/'
                os.makedirs(dest, exist_ok=True)
                O = (624 - 512) // 2
                S = 512
                for p in tqdm(sorted(glob(J(src, '*.jpg')))):
                    name = os.path.basename(p)
                    i = Image.open(p)
                    i2 = i.crop((O, O, S+O, S+O))
                    i2.save(J(dest, name))

    class ViewArgs(BaseMLCLI.CommonArgs):
        label: str = '0001'

    def run_view(self, a):
        image = Image.open(f'data/images/{a.label}.jpg')
        label = f'data/labels/{a.label}.txt'
        bbs = read_bbs(label)

        SIZE = 624
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf')

        for bb in bbs:
            id = bb['id']
            x,y,w,h = bb['anchors']
            rect = np.array([x-w/2, y-h/2, x+w/2, y+h/2]) * SIZE
            rect = rect.astype(int)
            draw.rectangle(tuple(rect), outline='red')
            draw.text((rect[0], rect[1]), f'id:{id}')
        image.save('example.png')


    class FlipArgs(BaseMLCLI.CommonArgs):
        src: str

    def run_flip(self, a):
        df = pd.read_excel('data/table.xlsx', converters={'label': str})
        for i, row in df.iterrows():
            label = row['label']
            if row['test']:
                continue
            image = Image.open(J(a.src, f'images/{label}.jpg'))
            bbs = read_bbs(J(a.src, f'labels/{label}.txt'))

            flipped = ImageOps.mirror(image)

            for i in range(6):
                assert bbs[i]['id'] == str(i)

            bbs[0]['id'] = 3
            bbs[1]['id'] = 4
            bbs[2]['id'] = 5
            bbs[3]['id'] = 0
            bbs[4]['id'] = 1
            bbs[5]['id'] = 2

            os.makedirs('tmp/flip/images', exist_ok=True)
            os.makedirs('tmp/flip/labels', exist_ok=True)
            flipped.save(f'tmp/flip/images/f{label}.jpg')
            write_bbs(bbs, f'tmp/flip/labels/f{label}.txt')


    class SplitFoldArgs(BaseMLCLI.CommonArgs):
        count: int
        flip: bool = Field(False, cli=('--flip', ))

    def run_split_fold(self, a):
        df = pd.read_excel('data/tables/main.xlsx', converters={'label': str})

        skf = StratifiedKFold(n_splits=a.count, shuffle=True, random_state=a.seed)
        skf.get_n_splits(df, df['treatment'])
        x = df['label']
        y = df['treatment']
        for i, (train_index, test_index) in enumerate(skf.split(x, y)):
            num = i + 1
            ii = np.concatenate([test_index, train_index])
            for j, i in enumerate(ii):
                is_test = j < len(test_index)
                row = df.iloc[i]
                label = row['label']
                src_image = f'data/images/{label}.jpg'
                src_label = f'data/labels/{label}.txt'
                t = 'test' if is_test else 'train'
                subdir = f'data/yolo/fold{num}/{t}'
                os.makedirs(f'{subdir}/images', exist_ok=True)
                os.makedirs(f'{subdir}/labels', exist_ok=True)
                dest_image = f'{subdir}/images/{label}.jpg'
                dest_label = f'{subdir}/labels/{label}.txt'
                shutil.copyfile(src_image, dest_image)
                shutil.copyfile(src_label, dest_label)

                if a.flip and not is_test:
                    bbs = read_bbs(src_label)
                    flipped = ImageOps.mirror(Image.open(src_image))
                    for i in range(6):
                        assert bbs[i]['id'] == str(i)
                    bbs[0]['id'] = 3
                    bbs[1]['id'] = 4
                    bbs[2]['id'] = 5
                    bbs[3]['id'] = 0
                    bbs[4]['id'] = 1
                    bbs[5]['id'] = 2
                    flipped.save(f'{subdir}/images/f{label}.jpg')
                    write_bbs(bbs, f'{subdir}/labels/f{label}.txt')

    class MapArgs(BaseMLCLI.CommonArgs):
        size: str = 's'
        nosort: bool = Field(False, cli=('--nosort', ))

    def run_map(self, a):
        # DIR = 'data/yolo_detects_6folds/labels'
        DIR = f'yolov5/runs/detect/{a.size}/labels'

        counts = []
        gt_dfs = {}

        metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=7)

        total = 0
        correct_50 = 0

        for p in sorted(glob('data/labels/0*.txt')):
            id = os.path.splitext(os.path.basename(p))[0]
            df = read_label_as_df(p)
            gt_dfs[id] = df

        for p in sorted(glob(f'{DIR}/*.txt')):
            id = os.path.splitext(os.path.basename(p))[0]
            df = read_label_as_df(p, with_confidence=True)
            pairs = list(combinations(df.iterrows(), 2))
            intersection_count = 0

            total_center_x = np.mean(df[['x0', 'x1']].values.view())
            total_center_y = np.mean(df[['y0', 'y1']].values.view())

            if not a.nosort:
                # 1. drop by intersection > 0.3
                for pair in pairs:
                    index1, row1 = pair[0]
                    index2, row2 = pair[1]
                    iou = calc_iou(row1[['x0', 'y0', 'x1', 'y1']], row2[['x0', 'y0', 'x1', 'y1']])
                    if iou > 0.5:
                        # print(p, iou, row1['label'], row2['label'])
                        intersection_count += 1
                        center_x1 = (row1['x1'] + row1['x0']) / 2
                        center_y1 = (row1['y1'] + row1['y0']) / 2
                        center_x2 = (row2['x1'] + row2['x0']) / 2
                        center_y2 = (row2['y1'] + row2['y0']) / 2
                        len1 = (total_center_x - center_x1)**2 + (total_center_y - center_y1)**2
                        len2 = (total_center_x - center_x2)**2 + (total_center_y - center_y2)**2
                        idx =  index1 if len1 > len2 else index2
                        df.drop(idx, inplace=True)

                # 2. check label duplication
                ok = validate_bb(df)

                # 3. if not ok and extra ROIs
                if not ok and len(df) >= 6:
                    # 4. sort by distance to center
                    df['x_distance'] = 100
                    df['y_distance'] = 100
                    for idx, row in df.iterrows():
                        center_x = (row['x1'] + row['x0']) / 2
                        center_y = (row['y1'] + row['y0']) / 2
                        distance = (center_x - center_x1)**2 + (center_y - center_y1)**2
                        df.loc[idx, 'x_distance'] = center_x - total_center_x
                        df.loc[idx, 'y_distance'] = center_y - total_center_y
                        df.loc[idx, 'distance'] = distance

                    # df = df.sort_values('distance').iloc[:6].copy()

                    # 3. do re-labeling
                    # select left:
                    rows = df[df['x_distance'] < 0]
                    # select 3 nearest ROI
                    rows = df[df['distance'].isin(rows.sort_values('distance')['distance'].nsmallest(3))]

                    rows = rows.sort_values('y0')
                    # most top -> 1
                    df.at[rows.iloc[0].name, 'label'] = 1

                    rows = rows.sort_values('x0')
                    # most left -> 2
                    df.at[rows.iloc[0].name, 'label'] = 2
                    # most right -> 3
                    df.at[rows.iloc[-1].name, 'label'] = 3

                    # # select right:
                    # rows = df[df['x_distance'] > 0]
                    # rows = rows.sort_values('y0')
                    # df.at[rows.iloc[0].name, 'label'] = 4
                    # df.at[rows.iloc[1].name, 'label'] = 5
                    # df.at[rows.iloc[2].name, 'label'] = 6

            # endif not nosort

            # l = len(df)
            # if l != 6 or not ok:
            #     print(p, l)
            #     print(df)

            df = df.sort_values('label').copy()

            gt_df = gt_dfs[id]

            gt = gt_df[['x0', 'y0', 'x1', 'y1', 'label']].values
            # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
            gt = np.append(gt, np.zeros([6, 2]), axis=1)

            # [xmin, ymin, xmax, ymax, class_id, confidence]
            pred = df[['x0', 'y0', 'x1', 'y1', 'label', 'conf']].values
            # pred[:, -1] = 1.0
            metric_fn.add(pred, gt)


            for (bb1, bb2) in zip(gt, pred):
                bb1 = bb1[:4]
                bb2 = bb2[:4]
                iou = calc_iou(bb1, bb2)
                if iou > 0.5:
                    correct_50 += 1
                total += 1

        # plt.hist(counts)
        # plt.show()

        print(50, correct_50/total)

        print(f"VOC PASCAL mAP: {metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1))['mAP']}")
        print(f"VOC PASCAL mAP in all points: {metric_fn.value(iou_thresholds=0.5)['mAP']}")
        print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")

if __name__ == '__main__':
    cli = CLI()
    cli.run()
