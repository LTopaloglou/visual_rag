import os
from typing import List
from image_utils import ImageUtils
from retriever import Retriever
from vqa import VisualQA

class VisualRAGPipeline:
    
    def __init__(self, device: str = "cpu"):
        self.utils = ImageUtils()
        self.retriever = Retriever(device=device)
        self.vqa = VisualQA(device=device)

    def predict(self, image_directory: str, queries: List[str]) -> List[str]:
        images = []
        for path in os.listdir(image_directory):
            full_path = os.path.join(image_directory, path)
            full_res_image = self.utils.load_image(full_path)
            downsampled_image = self.utils.resize(full_res_image)
            images.append(downsampled_image)
        self.retriever.encode(images)
        retrieved_images = self.retriever.retrieve(queries)
        for image in retrieved_images:
            print(image)
        
        return None