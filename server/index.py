from ultralytics import YOLO
from flask import Flask, request
from PIL import Image
import uuid
import cv2
import random

app = Flask(__name__)


print("Loading model")
model = YOLO("./best.pt")
print("Model loaded")

colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(24)]


@app.route("/api/image", methods=["POST"])
def predictImage():
    image = request.files["img"]
    if image is None:
        return {"message": "No image received"}

    print("[*]Detecting Image")
    result = model([Image.open(image)], conf=0.6)[0]
    print("[.]Detection complete")

    data = result.boxes.data.cpu().tolist()
    h, w = result.orig_shape

    names = result.names

    r = []
    for row in data:
        box = [row[0] / w, row[1] / h, row[2] / w, row[3] / h]
        conf = row[4]
        classId = int(row[5])
        name = names[classId]
        r.append(
            {
                "box": box,
                "confidence": conf,
                "classId": classId,
                "name": name,
                "color": "#%02x%02x%02x" % tuple(colors[classId]),
            }
        )

    return {"frame": r}

@app.route("/")
def home():
    return {"message": "Moye moye"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=False)
