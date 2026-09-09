import cv2
import numpy as np
import pandas as pd


class KolamFeatureExtractor:
    """Extracts geometric symmetry, topological, and shape features

    from binarized Kolam pattern images.
    """

    def __init__(self, image_size=(256, 256)):
        self.image_size = image_size

    def preprocess_image(self, image_path):
        """Loads image, converts to grayscale, and applies Otsu thresholding."""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")

        img_resized = cv2.resize(img, self.image_size)

        # Binarize image (isolating white strokes on dark background)
        _, binary = cv2.threshold(
            img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    def calculate_symmetry(self, binary_img):
        """Computes bilateral (vertical/horizontal) and 90-degree rotational symmetry

        using Intersection over Union (IoU).
        """
        norm_img = binary_img / 255.0

        # Horizontal & Vertical flips
        h_flip = np.fliplr(norm_img)
        v_flip = np.flipud(norm_img)

        # Rotations
        rot_90 = np.rot90(norm_img)

        # IoU metrics
        v_sym = np.sum(norm_img * h_flip) / (
            np.sum(np.logical_or(norm_img, h_flip)) + 1e-6
        )
        h_sym = np.sum(norm_img * v_flip) / (
            np.sum(np.logical_or(norm_img, v_flip)) + 1e-6
        )
        r90_sym = np.sum(norm_img * rot_90) / (
            np.sum(np.logical_or(norm_img, rot_90)) + 1e-6
        )

        return {
            "vertical_symmetry": float(v_sym),
            "horizontal_symmetry": float(h_sym),
            "rotational_symmetry_90": float(r90_sym),
        }

    def extract_topological_features(self, binary_img):
        """Extracts contour counts, bounding box metrics, and Hu Moments."""
        contours, _ = cv2.findContours(
            binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        loop_count = len(contours)
        stroke_density = float(np.sum(binary_img / 255.0) / binary_img.size)

        # Hu Moments (invariant to scale, rotation, and translation)
        moments = cv2.moments(binary_img)
        hu_moments = cv2.HuMoments(moments).flatten()
        log_hu = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        features = {
            "loop_count": loop_count,
            "stroke_density": stroke_density,
        }

        # Add 7 Hu moments to feature dictionary
        for i, hu in enumerate(log_hu):
            features[f"hu_moment_{i+1}"] = float(hu)

        return features

    def extract_all(self, image_path):
        """Combines all feature sets into a single dictionary."""
        binary = self.preprocess_image(image_path)
        symmetry_feats = self.calculate_symmetry(binary)
        topo_feats = self.extract_topological_features(binary)

        combined = {**symmetry_feats, **topo_feats}
        return combined


if __name__ == "__main__":
    print("Feature extractor module ready. Import and use KolamFeatureExtractor.")
