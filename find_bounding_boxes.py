import numpy as np
import pandas as pd
from scipy import ndimage
import os
from params import args

MARGIN = 64


def find_slices(mask_img):
    mask = mask_img > 100
    label_im, nb_labels = ndimage.label(mask)
    # Find the largest connect component
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    mask_size = sizes < 50000
    remove_pixel = mask_size[label_im]
    label_im[remove_pixel] = 0
    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)
    # Now that we have only one connect component, extract it's bounding box
    slice_y, slice_x = ndimage.find_objects(label_im == 1)[0]
    return slice_x, slice_y


def find_bounding_boxes():
    img_width = args.img_width
    img_height = args.img_height
    masks_dir = args.pred_mask_dir
    boxes = process_images(img_height, img_width, masks_dir)
    df = pd.DataFrame(boxes)
    df.to_csv("boxes.csv", header=['filename', 'y_start', 'y_end', 'x_start', 'x_end'], index=False)


def process_images(img_height, img_width, masks_dir):
    boxes = []
    for i, filename in enumerate(sorted(os.listdir(masks_dir))):
        mask_img = ndimage.imread(os.path.join(masks_dir, filename), mode='L')
        expanded = np.zeros((1280, 1920), dtype=mask_img.dtype)
        expanded[:, 1:-1] = mask_img
        mask_img = expanded
        slice_x, slice_y = find_slices(mask_img)
        # we should expand by at least 32px + ceil to closest divisible 32
        x_start = max(slice_x.start - MARGIN, 0)
        x_end = min(slice_x.stop + MARGIN, img_width)
        y_start = max(slice_y.start - MARGIN, 0)
        y_end = min(slice_y.stop + MARGIN, img_height)

        bb_height = y_end - y_start
        bb_width = x_end - x_start

        if bb_width % MARGIN != 0:
            bb_width_expand = (bb_width // MARGIN + 1) * MARGIN
            x_start = min(x_start, max(0, x_start - MARGIN))
            x_end = x_start + bb_width_expand

        if bb_height % MARGIN != 0:
            bb_height_expand = (bb_height // MARGIN + 1) * MARGIN
            y_start = min(y_start, max(0, y_start - MARGIN))
            y_end = y_start + bb_height_expand
        assert (x_end - x_start) % MARGIN == 0
        assert (y_end - y_start) % MARGIN == 0

        boxes.append((filename[:-4] + ".jpg", y_start, y_end, x_start, x_end))
        if i % 100 == 0:
            print("processed {} images".format(i))
    return boxes


if __name__ == '__main__':
    find_bounding_boxes()
