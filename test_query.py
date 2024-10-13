import requests
import base64
import os
import random

from io import BytesIO
from pydantic import BaseModel
from PIL import Image


class FoundedObjectsSchema(BaseModel):
    name: str
    class_: int
    confidence: float
    min_coords: tuple[int, int]
    max_coords: tuple[int, int]


class ImageResponseSchema(BaseModel):
    is_detected: bool
    objects: list["FoundedObjectsSchema"]
    proceeded_image: str


URL = "http://176.109.100.140:8080/find_object/"


def send(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {"confidence": 0}

        response = requests.post(URL, files=files, data=data)

    data = ImageResponseSchema.model_validate_json(response.text)

    return data


images_paths = [
    (os.path.abspath(r"./helicopters"), "helicopter"),
    (os.path.abspath(r"./planes"), "plane")
]

file_paths = []

for path, type_ in images_paths:
    file_paths.extend((os.path.join(path, i), type_) for i in os.listdir(path))

random.shuffle(file_paths)

for path, type_ in file_paths:
    data = send(path)

    format_data = data.model_copy()
    format_data.proceeded_image = ""
    print(format_data.model_dump_json(indent=4))

    encoded = data.proceeded_image
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data))

    os.makedirs("./final", exist_ok=True)
    if type_ in [i.name for i in data.objects]:
        image.save("./final/"+os.path.split(path)[1])
