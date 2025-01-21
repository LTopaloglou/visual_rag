import os
import gc
from typing import List
from image_utils import ImageUtils
from retriever import Retriever
from vqa import VisualQA

class VisualRAGPipeline:
    
    def __init__(self, demo_mode: bool = False):
        #TODO: on garbage_collect mode, maybe only initialize reader right before reading (after retriever deallocated)
        self.demo = demo_mode
        self.utils = ImageUtils()
        self.retriever = Retriever()
        if not self.low_mem:
            self.vqa = VisualQA()

    def predict(self, image_directory: str, queries: List[str]) -> List[str]:
        images = []
        for path in os.listdir(image_directory):
            full_path = os.path.join(image_directory, path)
            full_res_image = self.utils.load_image(full_path)
            if self.demo:
                print(f"Size of image: {full_res_image.size[0]}, {full_res_image.size[1]}")
            downsampled_image = self.utils.resize(full_res_image)
            images.append(downsampled_image)
        self.retriever.encode(images)
        retrieved_images = self.retriever.retrieve(queries)
        if self.low_mem:
            self.retriever = None
            gc.collect()
            self.vqa = VisualQA
        if self.demo:
            for image in retrieved_images:
                image.show()
        
        return None