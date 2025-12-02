"""
Machine Learning logic for image classification and preprocessing.
"""

import io
import random
from typing import Tuple, List
import numpy as np
from PIL import Image


class ImagePredictor:
    """
    A class for performing image prediction and various preprocessing operations.

    This class encapsulates the logic for:
    - Simulating image classification predictions.
    - Resizing images.
    - Converting images to grayscale.
    - Normalizing image pixel values.
    - Cropping images.

    Attributes:
        class_names (List[str]): A list of class names used for prediction.
    """

    def __init__(self, class_names: List[str] = None):
        """
        Initialize the ImagePredictor instance.

        Args:
            class_names (List[str], optional): A custom list of class names to be used for
                prediction. If None, a default set of classes (cat, dog, etc.) is used.
        """
        if class_names is None:
            self.class_names = [
                "cat",
                "dog",
                "bird",
                "fish",
                "horse",
                "car",
                "bicycle",
                "airplane",
                "boat",
                "train",
            ]
        else:
            self.class_names = class_names

    def predict(  # pylint: disable=unused-argument
        self, image_path: str = None, seed: int = None
    ) -> dict:
        """
        Predict the class of an image.

        Currently, this method simulates a prediction by randomly selecting a class
        from the available class names and assigning a random confidence score.

        Args:
            image_path (str, optional): The file path to the image. Currently unused in the
                mock prediction.
            seed (int, optional): A seed for the random number generator to ensure
                reproducible results.

        Returns:
            dict: A dictionary containing:
                - 'predicted_class' (str): The name of the predicted class.
                - 'confidence' (float): The confidence score of the prediction (between 0.7 and
                  0.99).
                - 'all_classes' (List[str]): A list of all available class names.
        """
        if seed is not None:
            random.seed(seed)

        predicted_class = random.choice(self.class_names)
        confidence = round(random.uniform(0.7, 0.99), 2)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_classes": self.class_names,
        }

    def resize_image(
        self, image_path: str, width: int, height: int, output_path: str = None
    ) -> Tuple[int, int]:
        """
        Resize an image file to specified dimensions.

        Args:
            image_path (str): The path to the input image file.
            width (int): The desired width in pixels.
            height (int): The desired height in pixels.
            output_path (str, optional): The path to save the resized image. If None, the
                image is not saved.

        Returns:
            Tuple[int, int]: A tuple containing the new (width, height) of the resized image.
        """
        with Image.open(image_path) as img:
            resized_img = img.resize((width, height), Image.Resampling.LANCZOS)

            if output_path:
                resized_img.save(output_path)

            return resized_img.size

    def resize_image_from_bytes(
        self, image_bytes: bytes, width: int, height: int, image_format: str
    ) -> bytes:
        """
        Resize an image provided as bytes.

        This method is useful for processing images directly from memory (e.g., uploaded files)
        without saving them to disk first.

        Args:
            image_bytes (bytes): The raw image data.
            width (int): The desired width in pixels.
            height (int): The desired height in pixels.
            image_format (str): The format of the image (e.g., 'jpeg', 'png') to use for the output.

        Returns:
            bytes: The resized image data as bytes.
        """
        img = Image.open(io.BytesIO(image_bytes))
        resized_img = img.resize((width, height), Image.Resampling.LANCZOS)

        output_bytes = io.BytesIO()
        resized_img.save(output_bytes, format=image_format)
        return output_bytes.getvalue()

    def convert_to_grayscale(self, image_path: str, output_path: str = None) -> str:
        """
        Convert an image file to grayscale.

        Args:
            image_path (str): The path to the input image file.
            output_path (str, optional): The path to save the grayscale image. If None, the
                image is not saved.

        Returns:
            str: The mode of the converted image (usually 'L' for grayscale).
        """
        with Image.open(image_path) as img:
            grayscale_img = img.convert("L")

            if output_path:
                grayscale_img.save(output_path)

            return grayscale_img.mode

    def convert_to_grayscale_from_bytes(
        self, image_bytes: bytes, image_format: str
    ) -> bytes:
        """
        Convert an image provided as bytes to grayscale.

        Args:
            image_bytes (bytes): The raw image data.
            image_format (str): The format of the image (e.g., 'jpeg', 'png') to use for the output.

        Returns:
            bytes: The grayscale image data as bytes.
        """
        img = Image.open(io.BytesIO(image_bytes))
        grayscale_img = img.convert("L")
        output_bytes = io.BytesIO()
        grayscale_img.save(output_bytes, format=image_format)
        return output_bytes.getvalue()

    def normalize_image(self, image_path: str, output_path: str = None) -> dict:
        """
        Normalize an image file and optionally save it.

        Normalization involves calculating the mean and standard deviation of pixel values
        and scaling the image data.

        Args:
            image_path (str): The path to the input image file.
            output_path (str, optional): The path to save the normalized image. If None, the
                image is not saved.

        Returns:
            dict: A dictionary containing the image statistics:
                - 'mean': The mean pixel value.
                - 'std': The standard deviation of pixel values.
        """
        with Image.open(image_path) as img:
            image_format = img.format
            img_array = np.array(img).astype(np.float32)

            mean = np.mean(img_array, axis=(0, 1))
            std = np.std(img_array, axis=(0, 1))

            # Perform normalization for saving if output_path is provided
            if output_path:
                epsilon = 1e-6
                normalized_array = (img_array - mean) / (std + epsilon)
                min_val, max_val = np.min(normalized_array), np.max(normalized_array)
                if max_val - min_val > epsilon:
                    scaled_array = (
                        255 * (normalized_array - min_val) / (max_val - min_val)
                    )
                else:
                    scaled_array = np.zeros_like(normalized_array)
                scaled_array = scaled_array.astype(np.uint8)
                normalized_img = Image.fromarray(scaled_array)
                normalized_img.save(output_path, format=image_format)

            return {
                "mean": np.round(mean, 2).tolist(),
                "std": np.round(std, 2).tolist(),
            }

    def normalize_image_from_bytes(
        self, image_bytes: bytes, image_format: str
    ) -> bytes:
        """
        Normalize an image provided as bytes.

        This method performs contrast stretching to normalize the image for visualization purposes.
        It scales the pixel values to the full 0-255 range.

        Args:
            image_bytes (bytes): The raw image data.
            image_format (str): The format of the image (e.g., 'jpeg', 'png') to use for the output.

        Returns:
            bytes: The normalized image data as bytes.
        """
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img).astype(np.float32)

        # Use a small epsilon to avoid division by zero
        epsilon = 1e-6
        mean = np.mean(img_array, axis=(0, 1))
        std = np.std(img_array, axis=(0, 1))
        normalized_array = (img_array - mean) / (std + epsilon)

        # Scale to 0-255 for visualization as a standard image
        min_val, max_val = np.min(normalized_array), np.max(normalized_array)
        if max_val - min_val > epsilon:
            scaled_array = 255 * (normalized_array - min_val) / (max_val - min_val)
        else:
            scaled_array = np.zeros_like(normalized_array)
        scaled_array = scaled_array.astype(np.uint8)

        normalized_img = Image.fromarray(scaled_array)
        output_bytes = io.BytesIO()
        normalized_img.save(output_bytes, format=image_format)
        return output_bytes.getvalue()

    def crop_image(
        self,
        image_path: str,
        box: Tuple[int, int, int, int],
        output_path: str = None,
    ) -> Tuple[int, int]:
        """
        Crop an image file to a specified region.

        Args:
            image_path (str): The path to the input image file.
            box (Tuple[int, int, int, int]): A tuple defining the crop region (left, top,
                right, bottom).
            output_path (str, optional): The path to save the cropped image. If None, the
                image is not saved.

        Returns:
            Tuple[int, int]: A tuple containing the (width, height) of the cropped image.
        """
        with Image.open(image_path) as img:
            cropped_img = img.crop(box)

            if output_path:
                cropped_img.save(output_path)

            return cropped_img.size

    def crop_image_from_bytes(
        self, image_bytes: bytes, box: Tuple[int, int, int, int], image_format: str
    ) -> bytes:
        """
        Crop an image provided as bytes.

        Args:
            image_bytes (bytes): The raw image data.
            box (Tuple[int, int, int, int]): A tuple defining the crop region (left, top,
                right, bottom).
            image_format (str): The format of the image (e.g., 'jpeg', 'png') to use for the
                output.

        Returns:
            bytes: The cropped image data as bytes.
        """
        img = Image.open(io.BytesIO(image_bytes))
        cropped_img = img.crop(box)
        output_bytes = io.BytesIO()
        cropped_img.save(output_bytes, format=image_format)
        return output_bytes.getvalue()
