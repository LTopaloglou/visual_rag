from pipeline import VisualRAGPipeline

pipeline = VisualRAGPipeline(
    device="cuda:0"
    )
pipeline.predict("images/", "How many plates are in the photo with the yellow potatoes, tomato salad and fried calamari?")