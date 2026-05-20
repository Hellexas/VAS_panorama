#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import sys
import os

# ANSI color codes for pretty console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_info(msg):
    print(f"{Colors.OKBLUE}[INFO]{Colors.ENDC} {msg}")

def log_success(msg):
    print(f"{Colors.OKGREEN}[SUCCESS]{Colors.ENDC} {Colors.BOLD}{msg}{Colors.ENDC}")

def log_warning(msg):
    print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} {msg}")

def log_error(msg):
    print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {msg}", file=sys.stderr)

def detect_and_match_features(img1, img2, ratio_threshold=0.7):
    """
    Detect SIFT keypoints and descriptors in both images and match them.
    Filters matches using Lowe's ratio test.
    """
    log_info("Converting images to grayscale...")
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    log_info("Initializing SIFT detector...")
    sift = cv2.SIFT_create()

    log_info("Detecting keypoints and computing descriptors...")
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    log_info(f"Image 1: Found {len(kp1)} keypoints.")
    log_info(f"Image 2: Found {len(kp2)} keypoints.")

    if des1 is None or des2 is None:
        log_error("Could not compute descriptors. One of the images might lack texture.")
        return None, None, None, None

    log_info("Matching features using Brute-Force Matcher...")
    bf = cv2.BFMatcher()
    # knnMatch returns k best matches for each descriptor
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

    log_info(f"Found {len(good_matches)} good matches after Lowe's ratio test (threshold={ratio_threshold}).")
    return kp1, kp2, good_matches, (img1, img2)

def compute_homography(kp1, kp2, matches, min_matches=10):
    """
    Compute the Homography matrix using RANSAC.
    """
    if len(matches) < min_matches:
        log_error(f"Not enough matches to compute homography. Required: {min_matches}, Found: {len(matches)}")
        return None

    log_info("Extracting coordinates of matched keypoints...")
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    log_info("Computing Homography matrix using RANSAC...")
    # cv2.RANSAC finds the best perspective transformation while filtering out outliers
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    inliers = np.sum(mask)
    log_info(f"RANSAC inliers: {inliers} / {len(matches)} matches ({inliers/len(matches)*100:.1f}%)")

    return H

def blend_images(warped_img1, img2_translated, mask1, mask2):
    """
    Blend two images using a distance-transform-based feathering technique.
    Provides a smooth, seamless transition in the overlapping region.
    """
    log_info("Calculating overlap region and blending weights...")
    
    # Intersection of both masks is the overlap region
    overlap = cv2.bitwise_and(mask1, mask2)
    
    if np.sum(overlap) == 0:
        log_warning("No overlap detected. Performing a simple direct merge.")
        # Direct merge
        result = np.where(img2_translated > 0, img2_translated, warped_img1)
        return result

    # Compute distance transforms to find distance to the boundaries of masks
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 3)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 3)

    # Normalize distances in the overlap region
    dist_sum = dist1 + dist2
    # Avoid division by zero
    dist_sum[dist_sum == 0] = 1e-5

    w1 = dist1 / dist_sum
    w2 = 1.0 - w1

    # Expand dimensions for channel multiplication
    w1_3d = np.expand_dims(w1, axis=2)
    w2_3d = np.expand_dims(w2, axis=2)

    # Build the final output image
    result = np.zeros_like(warped_img1)
    
    # 1. Areas where only warped_img1 exists
    mask_only1 = (mask1 == 1) & (mask2 == 0)
    result[mask_only1] = warped_img1[mask_only1]
    
    # 2. Areas where only img2 exists
    mask_only2 = (mask2 == 1) & (mask1 == 0)
    result[mask_only2] = img2_translated[mask_only2]
    
    # 3. Overlap area blended using weights
    mask_overlap = (overlap > 0)
    blended_overlap = w1_3d * warped_img1 + w2_3d * img2_translated
    result[mask_overlap] = blended_overlap[mask_overlap].astype(np.uint8)

    return result

def stitch_images(img1, img2, H, blend=True):
    """
    Warp img1 using Homography H and stitch it together with img2.
    Translates coordinates to prevent cropping of negative coordinates.
    """
    log_info("Calculating canvas size to prevent cropping...")
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Corners of img1: [x, y]
    corners_1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    # Project corners of img1 to img2's perspective
    warped_corners_1 = cv2.perspectiveTransform(corners_1, H)
    
    # Corners of img2
    corners_2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    # Combine all corners to find bounding box of the canvas
    all_corners = np.concatenate((warped_corners_1, corners_2), axis=0)
    x_min, y_min = all_corners.min(axis=0).ravel()
    x_max, y_max = all_corners.max(axis=0).ravel()

    # Determine dimensions of the output canvas
    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))
    
    log_info(f"Target panorama dimensions: {canvas_w}x{canvas_h}")

    # Compute translation offsets
    tx = -x_min
    ty = -y_min

    # Create translation matrix
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    
    # Adjusted homography including translation
    H_translated = T.dot(H)

    log_info("Warping Image 1...")
    warped_img1 = cv2.warpPerspective(img1, H_translated, (canvas_w, canvas_h))

    log_info("Translating Image 2...")
    img2_translated = np.zeros_like(warped_img1)
    # Place img2 onto the translated position
    img2_translated[int(ty):int(ty)+h2, int(tx):int(tx)+w2] = img2

    # Masks for both warped/translated images
    # We convert to grayscale to find non-black pixel masks
    gray_warped1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY)
    mask1 = (gray_warped1 > 0).astype(np.uint8)

    gray_translated2 = cv2.cvtColor(img2_translated, cv2.COLOR_BGR2GRAY)
    mask2 = (gray_translated2 > 0).astype(np.uint8)

    if blend:
        stitched = blend_images(warped_img1, img2_translated, mask1, mask2)
    else:
        log_info("Blending disabled. Performing direct overwrite.")
        stitched = warped_img1.copy()
        # Direct copy of img2 on top of img1 where img2 has pixels
        stitched[mask2 == 1] = img2_translated[mask2 == 1]

    # Crop trailing black borders if any
    # Find bounding box of the non-zero pixels in stitched
    gray_stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray_stitched)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        stitched = stitched[y:y+h, x:x+w]
        log_info(f"Cropped black outer borders. Final size: {w}x{h}")

    return stitched

def main():
    parser = argparse.ArgumentParser(description="Stitch two overlapping images into a panorama using SIFT.")
    parser.add_argument("--img1", type=str, default="para11.jpg", help="Path to the first image (will be warped).")
    parser.add_argument("--img2", type=str, default="para12.jpg", help="Path to the second image (anchor image).")
    parser.add_argument("--output", type=str, default="panorama.jpg", help="Path to save the stitched panorama.")
    parser.add_argument("--ratio", type=float, default=0.7, help="Lowe's ratio test threshold (default: 0.7).")
    parser.add_argument("--no-blend", action="store_true", help="Disable distance-based blending.")
    parser.add_argument("--save-matches", type=str, default=None, help="Save a visualization of feature matches to this path.")

    args = parser.parse_args()

    if not os.path.exists(args.img1):
        log_error(f"Image 1 not found at path: {args.img1}")
        sys.exit(1)
    if not os.path.exists(args.img2):
        log_error(f"Image 2 not found at path: {args.img2}")
        sys.exit(1)

    log_info(f"Loading images:\n  Image 1: {args.img1}\n  Image 2: {args.img2}")
    img1 = cv2.imread(args.img1)
    img2 = cv2.imread(args.img2)

    if img1 is None:
        log_error(f"Failed to load image 1: {args.img1}")
        sys.exit(1)
    if img2 is None:
        log_error(f"Failed to load image 2: {args.img2}")
        sys.exit(1)

    # 1. Feature detection & matching
    kp1, kp2, matches, images = detect_and_match_features(img1, img2, ratio_threshold=args.ratio)

    if matches is None or len(matches) < 4:
        log_error("Failed to find enough keypoint matches. Panorama cannot be created.")
        sys.exit(1)

    # Optional matches visualization
    if args.save_matches:
        log_info(f"Saving match visualization to: {args.save_matches}")
        matched_viz = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(args.save_matches, matched_viz)

    # 2. Homography computation
    H = compute_homography(kp1, kp2, matches)

    if H is None:
        log_error("Homography estimation failed.")
        sys.exit(1)

    # 3. Stitch & Warp
    blend_enabled = not args.no_blend
    panorama = stitch_images(img1, img2, H, blend=blend_enabled)

    # 4. Save result
    log_info(f"Saving panorama to: {args.output}")
    cv2.imwrite(args.output, panorama)
    log_success(f"Successfully stitched and saved to {args.output}!")

if __name__ == "__main__":
    main()
