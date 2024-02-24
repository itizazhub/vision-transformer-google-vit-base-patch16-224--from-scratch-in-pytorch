import onnxruntime
import pandas as pd
import numpy as np
from pathlib import Path
from config import config
from PIL import Image
import os
import glob
import warnings
warnings.filterwarnings('ignore')

onnx_model_path = Path(config.onnx_model_path).joinpath("model.onnx")
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# Prepare input data (adjust shape and data type as needed)
# input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
if not os.path.exists(Path(config.inference_images)):
    print("No inferece_images directory")
    print("please make directory inference_images and put test_image.jpg in that file to do inference")
else:
    all_images = Path(config.inference_images).glob('*')
    for image in all_images:
        image_path = Path(config.inference_images).joinpath(image)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_array = image_array / 255.0 
        image_array = image_array.astype(np.float32)
        input_data = np.expand_dims(image_array, axis=0)
        outputs = ort_session.run(None, {'input': input_data})

        df = pd.read_csv(Path(config.result_folder_path).joinpath("classes.csv"))
        output_data = outputs[0]
        print(output_data.shape)
