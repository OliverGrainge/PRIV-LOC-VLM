import base64
import os
from glob import glob
from io import BytesIO

import yaml
from PIL import Image


def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a Python dictionary.

    :param file_path: Path to the YAML file.
    :return: Dictionary representation of the YAML content.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(
                file
            )  # Use safe_load to avoid arbitrary code execution
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML file. {e}")
        return None


def pil_to_base64(img):
    """
    Converts a PIL Image object to a base64 encoded string.

    :param img: PIL Image object to be converted
    :return: Base64 encoded string representation of the image in PNG format
    """
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def read_img_pths(dataset_folder):
    """Find images within 'dataset_folder'.
    use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders might be slow.

    Parameters -> dataset_folder : str, folder containing JPEG images

    Returns -> images_paths : list[str], paths of JPEG images within dataset_folder
    """

    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    print(f"Searching test images in {dataset_folder}")
    images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
    if len(images_paths) == 0:
        raise FileNotFoundError(
            f"Directory {dataset_folder} does not contain any JPEG images"
        )
    return images_paths
