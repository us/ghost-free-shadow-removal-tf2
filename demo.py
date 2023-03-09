from __future__ import division
from deshadower import *
import argparse
import glob
import numpy as np
import os


# unrar on mac
# unrar x -o+ Samples.rar
def prepare_image_from_filename(image_filename):
    img = cv2.imread(image_filename, -1)
    return prep_image(img)


def prepare_image(img, test_w=-1, test_h=-1):
    if test_w > 0 and test_h > 0:
        img = cv2.resize(np.float32(img), (test_w, test_h), cv2.INTER_CUBIC)
    return img / 255.0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to folder containing the model", required=True)
    parser.add_argument("--vgg_19_path", help="path to vgg 19 path model", required=True)
    parser.add_argument("--input_dir", default='./Samples', help="path to sample images")
    parser.add_argument("--use_gpu", default=0, type=int, help="which gpu to use")
    parser.add_argument("--is_hyper", default=1, type=int, help="use hypercolumn or not")
    parser.add_argument("--result_dir", default='results', help="path to the result dir")

    ARGS = parser.parse_args()
    test_w, test_h = 640, 480

    deshadower = Deshadower(ARGS.model, ARGS.vgg_19_path, ARGS.use_gpu, ARGS.is_hyper)

    if not os.path.isdir(ARGS.result_dir):
        os.makedirs(ARGS.result_dir)

    for image_filename in glob.glob(ARGS.input_dir + '/*.png'):
        img = cv2.imread(image_filename, -1)
        orig_img_x, orig_img_y = img.shape[1], img.shape[0]
        test_w = int(img.shape[1] * test_h / float(img.shape[0]))
        img = prepare_image(img, test_w, test_h)
        oimg, mask = deshadower.run(img)
        if not os.path.isdir(ARGS.result_dir):
            os.makedirs(ARGS.result_dir)

        if ARGS.result_dir == ARGS.input_dir:
            mask_output_filename = os.path.join(ARGS.result_dir, os.path.basename(image_filename).split(".")[0] + "_mask." + os.path.basename(image_filename).split(".")[1])
        else:
            mask_output_filename = os.path.join(ARGS.result_dir, os.path.basename(image_filename)) 
        
        mask_output = cv2.resize(mask, (orig_img_x, orig_img_y), cv2.INTER_CUBIC)
        cv2.imwrite(mask_output_filename, mask)
        print(mask_output_filename, "created.")
        with Image.open(mask_output_filename) as img:
                # Resize image
                img_resized = img.resize((orig_img_x, orig_img_y))
                # Save resized image
                img_resized.save(mask_output_filename)
