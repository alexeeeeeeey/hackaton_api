import io
import torch
import base64
import warnings
import os
import asyncio
import uvicorn

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont
from pandas.core.frame import DataFrame

warnings.filterwarnings("ignore", category=FutureWarning)

app = FastAPI()

FONT_SIZE = 40
RECTANGLE_COLOR = "green"
TEXT_COLOR = "green"


model_path = Path(os.path.abspath('./yolo5s-e100-nofixu-ds_final.pt'))
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=model_path,
)
model.eval()


class FoundedObjectsSchema(BaseModel):
    name: str
    class_: int
    confidence: float
    min_coords: tuple[int, int]
    max_coords: tuple[int, int]


class ImageResponseSchema(BaseModel):
    is_detected: bool
    objects: list[FoundedObjectsSchema]
    proceeded_image: str


async def create_img(img: Image.Image, objects: list[FoundedObjectsSchema]):
    draw = ImageDraw.Draw(img)

    for obj in objects:
        draw.rectangle(
            [obj.min_coords, obj.max_coords],
            outline=RECTANGLE_COLOR,
            width=3,
        )
        text_coords = [obj.min_coords[0],
                       max(0, obj.min_coords[1] - FONT_SIZE)]
        draw.text(
            text_coords,
            obj.name,
            font=ImageFont.load_default(FONT_SIZE),
            fill=TEXT_COLOR,
        )

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

    return img_base64


@app.post("/find_object/")
async def upload_image(file: UploadFile = File(...),
                       confidence: int = Form(30)):
    loop = asyncio.get_running_loop()
    time_start = loop.time()

    image = Image.open(io.BytesIO(await file.read()))

    time_start_model = loop.time()

    results = model(image)

    print(f"Model - {int((loop.time() - time_start_model) * 1000)} ms",
          end=", ")

    founded_raw: DataFrame = results.pandas().xyxy[0]

    objects = [
        FoundedObjectsSchema(
            name=obj["name"],
            class_=int(obj["class"]),
            confidence=float(obj["confidence"]),
            min_coords=(int(obj["xmin"]), int(obj["ymin"])),
            max_coords=(int(obj["xmax"]), int(obj["ymax"])),
        ) for obj in founded_raw.to_dict(orient='records')
        if obj["confidence"] >= confidence / 100
    ]

    proceeded_img = await create_img(image, objects)

    response = ImageResponseSchema(
        is_detected=len(founded_raw) > 0,
        objects=objects,
        proceeded_image=proceeded_img
    )

    print(f"Total - {int((loop.time() - time_start) * 1000)} ms")
    return response

print("start app")
uvicorn.run(app, port=8080, host="0.0.0.0", log_level="error")
