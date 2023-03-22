import shutil

import typer
import os
from PIL import Image

app = typer.Typer()


@app.command()
def main(
        image_folder: str = typer.Argument(..., help="Path to image folder, which contains mask images in the "
                                                     "same. Script consider only files with _mask.png suffix in "
                                                     "the end"),
        output_folder: str = typer.Argument(..., help="Path to output folder"),
):
    counter = 0
    # create test_A, test_B, test_C folders to output folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'test_A'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'test_B'), exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'test_C'), exist_ok=True)

    for mask_file in os.listdir(image_folder):
        if mask_file.endswith('_mask.png') or mask_file.endswith('_mask.jpg'):
            mask_img = Image.open(os.path.join(image_folder, mask_file)).convert('L')
            if mask_img.getextrema() == (0, 0):
                counter += 1
                print(f"Mask file '{mask_file}' is empty")
                # copy not rename
                # remove _mask in the end of mask file name
                original_file = mask_file[:-9] + mask_file[-4:]

                shutil.copy(os.path.join(image_folder, original_file),
                            os.path.join(output_folder, 'test_C', original_file))
            else:
                original_file = mask_file[:-9] + mask_file[-4:]
                shutil.copy(os.path.join(image_folder, original_file),
                            os.path.join(output_folder, 'test_A', original_file))
                shutil.copy(os.path.join(image_folder, mask_file), os.path.join(output_folder, 'test_B', mask_file))


if __name__ == '__main__':
    typer.run(main)
