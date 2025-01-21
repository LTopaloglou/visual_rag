import torch
from typing import List
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.http import models
from colpali_engine.models import ColQwen2, ColQwen2Processor

class Retriever:

    def __init__(self):
        self.db = QdrantClient(":memory:")
        self.device = "cpu" #could do mps for my m1 vector chip, but not enough memory
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
        #TODO: doing 1 batch atm, but might be faster to do sequential since ton of memory pressure, or smaller batches
        model_input = self.processor.process_images(images).to(self.device)
        embeddings = self.embedding_model(**model_input)
        points = []
        #TODO: need this? multivector = embedding.cpu().float().numpy().tolist()
        for embedding in embeddings:
            points.append(
                models.PointStruct(
                    #dont think I need an ID
                    vector=embedding,
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
        embeddings = self.embedding_model(**model_input)
        #TODO: need this? multivector_query = embedding[0].cpu().float().numpy().tolist()
        search_result = self.db.query_points(
            collection_name="images",
            query=embeddings,
            limit=top_k,
            timeout=100,
        )
        #TODO: get just image payload from search result
        return search_result