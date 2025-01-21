import torch
import numpy
from typing import List
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colpali_engine.models import ColQwen2, ColQwen2Processor

class Retriever:

    def __init__(self, device: str):
        self.db = QdrantClient(":memory:")
        self.device = device
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

    def encode(self, images: List[Image.Image]):
        #Encodes images into DB
        model_input = self.processor.process_images(images).to(self.device)
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
                        "hello": "word" #TODO: put full-res image here
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
        #TODO: get just image payload from search result
        return search_result