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
        filepaths = []
        for path in os.listdir(image_directory):
            full_path = os.path.join(image_directory, path)
            filepaths.append(full_path)
            full_res_image = self.utils.load_image(full_path)
            downsampled_image = self.utils.resize(full_res_image)
            images.append(downsampled_image)
        self.retriever.encode(images, filepaths)

        retrieved_files = self.retriever.retrieve(queries)
        inference_images = []
        print(f"Retrieved images: {retrieved_files}")
        for filepath in retrieved_files:
            full_res_image = self.utils.load_image(filepath)
            downsampled_image = self.utils.resize(full_res_image)
            inference_images.append(downsampled_image)
        
        answers = self.vqa.predict(inference_images, queries)
        
        for answer in answers:
            print(answer)

        return answers