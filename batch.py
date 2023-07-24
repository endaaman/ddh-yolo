import os
import shutil

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from endaaman import BaseCLI


class CLI(BaseCLI):
    def run_copy(self, a):
        df = pd.read_excel('data/table.xlsx', converters={'label': str})
        for i, row in df.iterrows():
            label = row['label']
            src_image = f'data/images/{label}.jpg'
            src_label = f'data/labels/{label}.txt'
            subdir = 'test' if row['test'] else 'train'
            os.makedirs(f'data/yolo/{subdir}/images', exist_ok=True)
            os.makedirs(f'data/yolo/{subdir}/labels', exist_ok=True)
            dest_image = f'data/yolo/{subdir}/images/{label}.jpg'
            dest_label = f'data/yolo/{subdir}/labels/{label}.txt'
            shutil.copyfile(src_image, dest_image)
            shutil.copyfile(src_label, dest_label)

    def run_view(self, a):
        image = Image.open('data/images/0001.jpg')
        label = 'data/labels/0001.txt'

        with open(label, 'r') as f:
            output = f.read()

        SIZE = 624
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype('/usr/share/fonts/ubuntu/Ubuntu-R.ttf')

        for line in output.split('\n'):
            tokens = line.split(' ')
            if len(tokens) != 5:
                continue
            id = tokens[0]
            anchors = [int(float(v)*SIZE) for v in tokens[1:]]
            # bbs.append({
            #     'id': tokens[0],
            #     'anchors': ,
            # })
            print(anchors)
            x,y,w,h = anchors
            rect = (x-w//2, y-h//2, x+w//2, y+h//2)
            draw.rectangle(rect, outline='red')
            draw.text((rect[0], rect[1]), f'id:{id}')
        image.save('bb.png')


if __name__ == '__main__':
    cli = CLI()
    cli.run()
