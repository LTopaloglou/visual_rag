import torch
import numpy
from typing import List
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colpali_engine.models import ColQwen2, ColQwen2Processor

class Retriever:

    def __init__(self, device: str, batch_size: int = 4):
        self.db = QdrantClient(":memory:")
        self.device = device
        self.batch_size = batch_size
        self.model_name = "vidore/colqwen2-v1.0-merged"
        self.processor = ColQwen2Processor.from_pretrained(self.model_name)
        self.embedding_model = ColQwen2.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self.db.create_collection(
            collection_name="images",
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                )
            )
        )

    def encode(self, images: List[Image.Image], filepaths: List[str]):
        #Encodes images into DB
        for index in range(0, len(images), self.batch_size):
            end_index = min(len(images), index + self.batch_size)
            image_batch = images[index:end_index]
            model_input = self.processor.process_images(image_batch).to(self.device)
            with torch.no_grad():
                embeddings = self.embedding_model(**model_input)
            points = []
            vectors = embeddings.cpu().float().numpy().tolist()
            for i, vector in enumerate(vectors):
                points.append(
                    models.PointStruct(
                        id=i,
                        vector=vector,
                        payload={
                            "filepath": filepaths[index + i]
                        }
                    )
                )
            self.db.upsert(
                collection_name="images",
                points=points
            )

    def retrieve(self, queries: List[str], top_k: int = 3) -> List[Image.Image]:
        #Retrieves from DB based on queries
        model_input = self.processor.process_queries(queries).to(self.device)
        with torch.no_grad():
            embeddings = self.embedding_model(**model_input)
        vectors = embeddings[0].cpu().float().numpy().tolist() #embeddings or embeddings[0]???
        search_result = self.db.query_points(
            collection_name="images",
            query=vectors,
            limit=top_k,
            timeout=100,
        )
        retrieved_files = [point.payload["filepath"] for point in search_result.points]
        return retrieved_files