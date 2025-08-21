import os
import cv2
import numpy as np

def find_best_match(sample_path, db_path, min_match_count=10):
    sample = cv2.imread(sample_path, cv2.IMREAD_GRAYSCALE)
    if sample is None:
        print(f"Error loading sample image: {sample_path}")
        return

    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(sample, None)

    best_score = 0
    best_match_file = None
    best_image = None
    best_kp1 = best_kp2 = best_mp = None

    for file in os.listdir(db_path)[:1000]:
        fingerprint_image_path = os.path.join(db_path, file)
        fingerprint_image = cv2.imread(fingerprint_image_path, cv2.IMREAD_GRAYSCALE)
        if fingerprint_image is None:
            continue

        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_image, None)

        if descriptors_2 is None:
            continue

        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 10}, {})
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        # Store all the good matches as per Lowe's ratio test.
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > min_match_count:
            src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Compute Homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # Calculate score as the ratio of inliers
            score = float(sum(matchesMask)) / len(matchesMask) * 100

            if score > best_score:
                best_score = score
                best_match_file = file
                best_image = fingerprint_image
                best_kp1, best_kp2, best_mp = keypoints_1, keypoints_2, good_matches

    if best_match_file:
        print(f"BEST MATCH: {best_match_file}")
        print(f"SCORE: {best_score:.2f}%")

        # Draw only inliers
        draw_params = dict(matchColor=(0, 255, 0),  # Draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # Draw only inliers
                           flags=2)

        result = cv2.drawMatches(sample, best_kp1, best_image, best_kp2, best_mp, None, **draw_params)
        result = cv2.resize(result, None, fx=2, fy=2)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No match found or not enough matches are found - %d/%d" % (len(good_matches), min_match_count))

# Usage
find_best_match(r"SOCOFing/Altered/Altered-Medium/3__M_Left_little_finger_Zcut.BMP", "SOCOFing/Real")
