def preprocess_image(image_path, image_size=(224, 224)):
    from PIL import Image
    import numpy as np

    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize(image_size)
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension
    return image_array