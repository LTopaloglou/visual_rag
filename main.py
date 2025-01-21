from pipeline import VisualRAGPipeline

pipeline = VisualRAGPipeline(
    low_mem=True,
    demo_mode=True
    )
pipeline.predict("images/", "How many plates are in the photo with the yellow potatoes, tomato salad and fried calamari?")