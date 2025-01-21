from PIL import Image

class ImageUtils:

    def resize(self, image: Image.Image, max_dim: int = 400) -> Image:
        #Resizes images for faster embedding
        original_width = image.size[0]
        original_height = image.size[1]
        resize_ratio = min(original_width/max_dim, original_height/max_dim)
        new_width = int(original_width / resize_ratio)
        new_height = int(original_height / resize_ratio)
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)
        return resized_image
    
    def load_image(self, path: str) -> Image.Image:
        return Image.open(path)