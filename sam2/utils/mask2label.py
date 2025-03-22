import json
import numpy as np

def convert_to_native_types(obj):
    if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
        return int(obj) if isinstance(obj, (np.int64, np.int32)) else float(obj)
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    else:
        return obj

def convert(group_id, masks):
    shapes = []

    for i in range(masks.shape[0]):
        # Example binary mask (replace this with your actual mask)
        binary_mask = np.squeeze(masks[i])

        # Step 1: Find the coordinates of all pixels with value 1
        coordinates = np.argwhere(binary_mask == 1)

        # Step 2: Convert to the required format
        segmentation = coordinates[:, [1, 0]].tolist()  # Swap columns to get [x, y] format

        # Create a new dictionary for each mask
        shape_format = {
            "label": "zebra",
            "score": None,
            "points": segmentation,  # Assign segmentation data to "points"
            "group_id": group_id[i],  # Assign group_id
            "description": "",
            "difficult": False,
            "shape_type": "polygon",
            "flags": {},
            "attributes": {},
            "kie_linking": []
        }

        # Convert NumPy types to native Python types
        shape_format = convert_to_native_types(shape_format)

        # Append the shape to the list
        shapes.append(shape_format)

    return shapes