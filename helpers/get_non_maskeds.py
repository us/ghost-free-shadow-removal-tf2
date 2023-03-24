import os
import shutil

import typer
from PIL import Image

app = typer.Typer()


@app.command()
def main(
        image_folder: str = typer.Argument(..., help="Path to image folder, which contains mask images in the "
                                                     "same. Script consider only files with _mask.png suffix in "
                                                     "the end"),
        output_folder: str = typer.Argument(..., help="Path to output folder"),
        get_only_empty_masks: bool = typer.Option(False, help="If True, script will copy only empty masks to "
                                                              "output folder. If False, script will copy only "
                                                              "non-empty masks to output folder")
):
    counter = 0
    # create test_A, test_B, test_C folders to output folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'train_A'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'train_B'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'train_C'), exist_ok=True)

    for mask_file in os.listdir(image_folder):
        if mask_file.endswith('_mask.png') or mask_file.endswith('_mask.jpg'):
            mask_img = Image.open(os.path.join(image_folder, mask_file)).convert('L')
            if mask_img.getextrema() == (0, 0):
                counter += 1
                print(f"Mask file '{mask_file}' is empty")
                # copy not rename
                # remove _mask in the end of mask file name
                original_file = mask_file[:-9] + mask_file[-4:]
                # if get_only_empty_masks:
                #     print("Saving to ", os.path.join(output_folder, original_file))
                #     shutil.copy(os.path.join(image_folder, original_file),
                #                 os.path.join(output_folder, "free_images", original_file))
                #     continue
                shutil.copy(os.path.join(image_folder, original_file),
                            os.path.join(output_folder, 'train_C', original_file))
            else:
                # if get_only_empty_masks:
                #     continue
                original_file = mask_file[:-9] + mask_file[-4:]
                shutil.copy(os.path.join(image_folder, original_file),
                            os.path.join(output_folder, 'train_A', original_file))
                shutil.copy(os.path.join(image_folder, mask_file), os.path.join(output_folder, 'train_B', mask_file))


if __name__ == '__main__':
    typer.run(main)
