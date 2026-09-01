"""Face-verification helpers used by the Streamlit application."""

from __future__ import annotations

import numpy as np
from deepface import DeepFace


def is_reference_person(person_crop: np.ndarray, reference_image: np.ndarray) -> bool:
    """Return whether a detected person contains the uploaded reference face."""
    try:
        result = DeepFace.verify(
            img1_path=person_crop,
            img2_path=reference_image,
            detector_backend="opencv",
            enforce_detection=True,
            silent=True,
        )
        return bool(result.get("verified", False))
    except (ValueError, TypeError):
        # A person bounding box does not always contain a detectable frontal face.
        return False
