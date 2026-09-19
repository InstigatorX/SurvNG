"""Bounded ORB registration shared by native evidence and camera calibration."""
import cv2
import numpy as np

def estimate_stream_alignment(live: np.ndarray, main: np.ndarray) -> tuple[float, float, float, float] | None:
    if live.size == 0 or main.size == 0:
        return None
    def gray(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = min(1.0, 640.0 / max(height, width))
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        return resized if resized.ndim == 2 else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    live_gray, main_gray = gray(live), gray(main)
    orb = cv2.ORB_create(nfeatures=600)
    live_keypoints, live_descriptors = orb.detectAndCompute(live_gray, None)
    main_keypoints, main_descriptors = orb.detectAndCompute(main_gray, None)
    if live_descriptors is None or main_descriptors is None:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(live_descriptors, main_descriptors)
    matches = sorted(matches, key=lambda item: item.distance)[:80]
    if len(matches) < 18:
        return None
    source = np.float32([live_keypoints[item.queryIdx].pt for item in matches])
    target = np.float32([main_keypoints[item.trainIdx].pt for item in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if matrix is None or inliers is None or int(inliers.sum()) < 18:
        return None
    # Only accept scale/translation: this configuration model deliberately
    # does not represent rotation, shear, or a changed perspective.
    if abs(float(matrix[0, 1])) > 0.02 or abs(float(matrix[1, 0])) > 0.02:
        return None
    live_h, live_w = live_gray.shape[:2]
    main_h, main_w = main_gray.shape[:2]
    return (
        float(matrix[0, 0]) * live_w / main_w,
        float(matrix[1, 1]) * live_h / main_h,
        float(matrix[0, 2]) / main_w,
        float(matrix[1, 2]) / main_h,
    )
