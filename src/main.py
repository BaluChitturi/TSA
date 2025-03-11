import os
import numpy as np
from src.feature_extractor import ghostnet_feature_extractor
from src.image_preprocessing import preprocess_image

def main():
    input_dir = 'E:/preprocessed_dataset'  # Update with your dataset path
    output_feature_file = 'features.npy'
    output_label_file = 'labels.npy'
    output_label_map = 'label_map.json'

    # Load the feature extractor model
    ghostnet_model = ghostnet_feature_extractor()

    # Feature extraction
    features = []
    labels = []
    label_map = {}
    label_index = 0

    for root, subdirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(root, file)

                # Extract class label from folder name
                relative_path = os.path.relpath(root, input_dir)
                if relative_path not in label_map:
                    label_map[relative_path] = label_index
                    label_index += 1

                # Load and preprocess image
                image_array = preprocess_image(image_path)
                image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

                # Extract feature vector using GhostNet
                feature_vector = ghostnet_model.predict(image_array, verbose=0)

                # Store feature and label
                features.append(feature_vector.flatten())  # Flatten to 1D
                labels.append(label_map[relative_path])

    # Convert to numpy arrays and save
    np.save(output_feature_file, np.array(features))
    np.save(output_label_file, np.array(labels))

    # Save label map correctly as JSON
    with open(output_label_map, "w") as json_file:
        json.dump(label_map, json_file)

    print(f"✅ Feature extraction complete! Features saved to {output_feature_file}, Labels saved to {output_label_file}")
    print(f"Label Mapping saved to {output_label_map}: {label_map}")

if __name__ == "__main__":
    main()