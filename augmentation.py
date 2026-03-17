import os
import cv2
import numpy as np
from PIL import Image

INPUT_FOLDERS = [
    "google_docs_pdfs_png",
    "python_pdfs_png",
    "word_pdfs_png"
]

OUTPUT_ROOT = "augmented_images"


def gaussian_noise(img):
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    return cv2.add(img, noise)


def jpeg_compression(img, output_path):
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 45])


def downsample(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2))
    return cv2.resize(small, (w, h))


def random_crop(img):
    h, w = img.shape[:2]
    crop_h = int(h * 0.98)
    crop_w = int(w * 0.98)

    start_x = np.random.randint(0, w - crop_w + 1)
    start_y = np.random.randint(0, h - crop_h + 1)

    cropped = img[start_y:start_y + crop_h, start_x:start_x + crop_w]
    return cv2.resize(cropped, (w, h))


def bit_depth(img):
    return (img // 64) * 64


def process_folder(folder):
    input_path = folder
    output_path = os.path.join(OUTPUT_ROOT, folder)

    os.makedirs(output_path, exist_ok=True)

    for file in os.listdir(input_path):
        if not file.endswith(".png"):
            continue

        path = os.path.join(input_path, file)
        img = cv2.imread(path)

        base = file.replace(".png", "")

        # 1 Noise
        cv2.imwrite(
            os.path.join(output_path, f"{base}__noise.png"),
            gaussian_noise(img)
        )

        # 2 Compression (jpeg)
        jpeg_compression(
            img,
            os.path.join(output_path, f"{base}__jpeg.jpg")
        )

        # 3 Downsample
        cv2.imwrite(
            os.path.join(output_path, f"{base}__downsample.png"),
            downsample(img)
        )

        # 4 Crop
        cv2.imwrite(
            os.path.join(output_path, f"{base}__crop.png"),
            random_crop(img)
        )

        # 5 Bit depth
        cv2.imwrite(
            os.path.join(output_path, f"{base}__bitdepth.png"),
            bit_depth(img)
        )


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for folder in INPUT_FOLDERS:
        print("Processing:", folder)
        process_folder(folder)


if __name__ == "__main__":
    main()