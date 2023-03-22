import cv2
import typer
import os


def widen_shadow(mask, pixel=5):
    # Structuring element filtresini oluşturun
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixel, pixel))

    # Maskenizi genişletin
    maske_genisletilmis = cv2.dilate(mask, structuring_element)

    # Genişletilmiş maskeyi döndürün
    return maske_genisletilmis


app = typer.Typer()


@app.command()
def main(
        mask_folder_path: str = typer.Argument(..., help="Path to image folder, which contains mask images in the "
                                                         "same. Script consider only files with _mask.png suffix in "
                                                         "the end"),
        output_folder_path: str = typer.Argument(..., help="Path to output folder"),
        pixel: int = typer.Option(100, help="Pixel size for structuring element"),
):
    os.makedirs(output_folder_path, exist_ok=True)
    for mask_file in os.listdir(mask_folder_path):
        if mask_file.endswith("_mask.png"):
            mask = cv2.imread(os.path.join(mask_folder_path, mask_file), cv2.IMREAD_GRAYSCALE)
            mask = widen_shadow(mask, pixel)
            cv2.imwrite(os.path.join(output_folder_path, mask_file), mask)


if __name__ == "__main__":
    typer.run(main)
