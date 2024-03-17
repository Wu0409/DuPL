import numpy as np
from PIL import Image
import os


def colorful(out, name):
    arr = out.astype(np.uint8)
    im = Image.fromarray(arr)

    palette = []
    for i in range(256):
        palette.extend((i, i, i))

    palette[:3 * 21] = np.array([
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128]
    ], dtype='uint8').flatten()

    im.putpalette(palette)
    im.save(target_dir + name)


if __name__ == '__main__':
    # VOC 21k
    # dir = './work_dir_voc_wseg/2023-09-28-22-37-08-148406_train_final_voc/segs/seg_preds/test/'
    # target_dir = './work_dir_voc_wseg/2023-09-28-22-37-08-148406_train_final_voc_21k/segs/convert/test/'

    # VOC
    dir = './work_dir_voc_wseg/2023-09-26-21-50-19-319439_train_voc/segs/seg_preds/test/'  # an example
    target_dir = './work_dir_voc_wseg/2023-09-26-21-50-19-319439_train_voc/segs/convert/test/'

    name_list = os.listdir(dir)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for i in name_list:
        img = np.array(Image.open(dir + i))
        colorful(img, i)
