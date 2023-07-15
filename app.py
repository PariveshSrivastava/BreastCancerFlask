import yolov5
import base64
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
from io import BytesIO
# from pymongo import MongoClient

model = yolov5.load('best.pt')
model.conf = 0.6

app = Flask(__name__)
# Allow requests from all routes
CORS(app)


colours = {0: 'blue', 1: 'green', 2: 'orange', 3: 'red'}


@app.route('/api/yolov5', methods=['POST'])
def process_image():
    mfa = request.headers.get('mfa')

    if mfa:
        try:
            ext = request.files['image'].filename.split('.')[-1]
            image = Image.open(request.files['image'])
            p = model(image, size=1280).pred[0]
            n = 0
            w = 0
            m = 0
            s = 0
            draw = ImageDraw.Draw(image)
    
            for box in p:
                x1, y1, x2, y2 = box[:4]
                if box[5] == 0:
                    n += 1
                elif box[5] == 1:
                    w += 1
                elif box[5] == 2:
                    m += 1
                elif box[5] == 3:
                    s += 1
                # Draw the bounding box rectangle
                draw.rectangle([(x1, y1), (x2, y2)],
                               outline=colours.get(int(box[5])), width=1)
    
            image_io = BytesIO()
            image.save(image_io, format=ext)
            image_io.seek(0)
            encoded_image = base64.b64encode(image_io.getvalue()).decode('utf-8')
    
            # Store image data in MongoDB collection
            data = {
                'image': encoded_image,
                'converted_image': None  # Placeholder for converted image, to be updated later
            }
    
            response = {
                'code': 0,
                'image': encoded_image,
                'n': n,
                'w': w,
                'm': m,
                's': s,
                'all': len(p),
                # 'inserted_id': str(inserted_id)  # Pass the inserted ID to the frontend for reference
            }
            return response
        except Exception as e:
            print(e)
            return {'code': 2}
    else:
        return {'code': 1}


if __name__ == '__main__':
    app.run(debug=True, port=5001)
    print("flask yoloV5 running at 5001")
