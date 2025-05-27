import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Paths
image_dir = "SegmentationTraining/images/train"
label_dir = "SegmentationTraining/labels/train"

for i in range(10):
    image_path = os.path.join(image_dir, f"synthetic_{i:04d}.jpg")
    label_path = os.path.join(label_dir, f"synthetic_{i:04d}.txt")

    print(f"Testing: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load {image_path}")
        continue

    height, width, _ = image.shape

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                cls = int(parts[0])
                coords = parts[1:]

                # Ensure coords are in pairs
                if len(coords) % 2 != 0:
                    print(f"Invalid number of coordinates in {label_path}")
                    continue

                # Convert normalized coords to pixel coords
                points = []
                for j in range(0, len(coords), 2):
                    x = int(coords[j] * width)
                    y = int(coords[j + 1] * height)
                    points.append((x, y))

                # Draw polygon
                pts = cv2.polylines(image.copy(), [np.array(points, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
                for (x, y) in points:
                    cv2.circle(pts, (x, y), 4, (255, 0, 0), -1)  # Optional: mark corners
                cv2.putText(pts, str(cls), (points[0][0], points[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

                image = pts

    # Convert and show
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title(f"YOLO Segmentation: synthetic_{i:04d}.jpg")
    plt.show()
