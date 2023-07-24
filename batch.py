import os
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from sklearn.model_selection import StratifiedKFold
from pydantic import Field

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



class CLI(BaseMLCLI):
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
        count: int = 5
        flip: bool = Field(False, cli=('--flip', ))

    def run_split_fold(self, a):
        df = pd.read_excel('data/table.xlsx', converters={'label': str})

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



if __name__ == '__main__':
    cli = CLI()
    cli.run()
