from typing import List

import typer as typer
from PIL import Image
import os


def split_image(image_path, output_folder, size):
    """
    Split the input image into several images of the given size and save them to the output folder.
    The file names include the original size and position of the image fragment.
    """
    os.makedirs(output_folder, exist_ok=True)
    image = Image.open(image_path)
    width, height = image.size

    x_steps = width // size[0]
    y_steps = height // size[1]

    # create a new background image with black color
    background = Image.new("RGB", ((x_steps + 1) * size[0], (y_steps + 1) * size[1]), "black")
    background.paste(image, (0, 0))

    # split the new image
    for x in range(x_steps + 1):
        for y in range(y_steps + 1):
            box = (
                x * size[0],
                y * size[1],
                (x + 1) * size[0],
                (y + 1) * size[1],
            )
            fragment = background.crop(box)

            # if the fragment is not of the desired size, paste it on a new background image
            if fragment.size != size:
                new_fragment = Image.new("RGB", size, "black")
                new_fragment.paste(fragment, (0, 0))
                fragment = new_fragment

            filename = f"{width}x{height}_{box}_{os.path.basename(image_path)}"
            fragment.save(os.path.join(output_folder, filename))


def merge_images(files: List, output_path):
    """
    Merge the images in the input folder into one image and save it to the output path.
    The file names include the original size and position of the image fragment.
    """
    # get the list of image files in the input folder

    # get the original size and position of the first image fragment
    first_file = files[0]

    # split the file name by underscore and parenthesis
    parts = os.path.basename(first_file).split("_")

    # get the width and height from the second part
    width, height = map(int, parts[0].split("x"))

    # get the box coordinates from the third part
    # box = parts[-1].strip("().png")

    # create a new background image with black color
    background = Image.new("RGB", (width, height), "black")

    # paste each image fragment on the background according to its position
    for file in files:
        # split the file name by underscore and parenthesis
        parts = os.path.basename(file).split("_")

        # get the box coordinates from the third part
        box = parts[1].strip("()")

        # convert them to integers
        left, top, right, bottom = map(int, box.split(", "))

        fragment = Image.open(file)
        background.paste(fragment, (left, top))

    # crop the background to remove any black borders
    background = background.crop((0, 0, width, height))

    # save the merged image to the output path
    background.save(output_path)


app = typer.Typer()


@app.command()
def main(
        input_folder=typer.Argument(..., help="The input folder"),
        output_folder=typer.Argument(..., help="The output folder"),
        split_or_merge=typer.Argument(..., help="Split or merge, e.g. `split` or `merge`"),
        size=typer.Argument("256,256", help="The size of the image fragments, e.g. 512,512"),
):
    os.makedirs(output_folder, exist_ok=True)

    if split_or_merge == "split":
        for file in os.listdir(input_folder):
            if file.endswith(".png") or file.endswith(".jpg"):
                split_image(os.path.join(input_folder, file), output_folder, tuple(map(int, size.split(","))))
    elif split_or_merge == "merge":
        all_files = os.listdir(input_folder)
        # get only png and jpg and jpeg files, eliminate others
        all_files = [file for file in all_files if
                     file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")]

        files_dict = {}
        for file in all_files:
            name = "_".join(file.split("_")[2:])
            if name in files_dict:
                files_dict[name].append(os.path.join(input_folder, file))
            else:
                files_dict[name] = [os.path.join(input_folder, file)]
        for filename, files in files_dict.items():
            merge_images(files, os.path.join(output_folder, filename))


if __name__ == '__main__':
    typer.run(main)
