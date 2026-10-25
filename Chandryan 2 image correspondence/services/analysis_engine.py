from pathlib import Path
import time

import cv2
import numpy as np


class LunarCorrespondenceEngine:

    def __init__(
        self,
        detector="SIFT",
        ratio_threshold=0.75,
        ransac_threshold=5.0,
        min_inliers=10
    ):
        self.detector_name = str(detector).upper()
        self.ratio_threshold = float(ratio_threshold)
        self.ransac_threshold = float(ransac_threshold)
        self.min_inliers = int(min_inliers)

        # Relaxed second pass for cross-sensor lunar imagery
        self.fallback_ratio = min(
            0.90,
            max(0.80, self.ratio_threshold + 0.10)
        )

    # =====================================================
    # IMAGE LOADING
    # =====================================================

    def _load_gray(self, path):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {path}"
            )

        return image

    def _load_color(self, path):

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise ValueError(
                f"Could not read image: {path}"
            )

        return image

    # =====================================================
    # PREPROCESSING
    # =====================================================

    def _preprocess(self, gray):

        gray = cv2.normalize(
            gray,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        clahe = cv2.createCLAHE(
            clipLimit=2.2,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        enhanced = cv2.GaussianBlur(
            enhanced,
            (3, 3),
            0
        )

        return enhanced

    # =====================================================
    # FEATURE DETECTOR
    # =====================================================

    def _create_detector(self):

        if self.detector_name == "ORB":

            return cv2.ORB_create(
                nfeatures=12000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                patchSize=31,
                fastThreshold=8
            )

        return cv2.SIFT_create(
            nfeatures=15000,
            nOctaveLayers=5,
            contrastThreshold=0.012,
            edgeThreshold=10,
            sigma=1.2
        )

    def _detect(self, image):

        detector = self._create_detector()

        keypoints, descriptors = (
            detector.detectAndCompute(
                image,
                None
            )
        )

        return keypoints or [], descriptors

    # =====================================================
    # MATCHING
    # =====================================================

    def _knn_ratio_matches(
        self,
        d1,
        d2,
        ratio
    ):

        if d1 is None or d2 is None:
            return []

        if len(d1) < 2 or len(d2) < 2:
            return []

        if self.detector_name == "ORB":

            matcher = cv2.BFMatcher(
                cv2.NORM_HAMMING,
                crossCheck=False
            )

        else:

            matcher = cv2.BFMatcher(
                cv2.NORM_L2,
                crossCheck=False
            )

        pairs = matcher.knnMatch(
            d1,
            d2,
            k=2
        )

        good = []

        for pair in pairs:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < ratio * n.distance:
                good.append(m)

        return good

    def _mutual_matches(
        self,
        source_desc,
        target_desc,
        ratio
    ):

        forward = self._knn_ratio_matches(
            source_desc,
            target_desc,
            ratio
        )

        backward = self._knn_ratio_matches(
            target_desc,
            source_desc,
            ratio
        )

        backward_pairs = {
            (m.queryIdx, m.trainIdx)
            for m in backward
        }

        mutual = []

        for m in forward:

            if (
                m.trainIdx,
                m.queryIdx
            ) in backward_pairs:

                mutual.append(m)

        return mutual, forward

    # =====================================================
    # POINT EXTRACTION
    # =====================================================

    def _points_from_matches(
        self,
        kp_source,
        kp_target,
        matches
    ):

        if not matches:
            return None, None

        source_points = np.float32([
            kp_source[m.queryIdx].pt
            for m in matches
        ]).reshape(-1, 1, 2)

        target_points = np.float32([
            kp_target[m.trainIdx].pt
            for m in matches
        ]).reshape(-1, 1, 2)

        return source_points, target_points

    # =====================================================
    # HOMOGRAPHY / RANSAC
    # =====================================================

    def _homography(
        self,
        source_points,
        target_points
    ):

        if (
            source_points is None
            or target_points is None
            or len(source_points) < 4
        ):
            return None, None

        try:

            H, mask = cv2.findHomography(
                source_points,
                target_points,
                cv2.RANSAC,
                self.ransac_threshold,
                maxIters=10000,
                confidence=0.995
            )

            return H, mask

        except Exception:

            return None, None

    # =====================================================
    # AFFINE FALLBACK
    # =====================================================

    def _affine_fallback(
        self,
        source_points,
        target_points
    ):

        if (
            source_points is None
            or target_points is None
            or len(source_points) < 3
        ):
            return None, None

        try:

            affine, mask = cv2.estimateAffinePartial2D(
                source_points.reshape(-1, 2),
                target_points.reshape(-1, 2),
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold,
                maxIters=10000,
                confidence=0.995,
                refineIters=20
            )

            if affine is None:
                return None, None

            H = np.vstack([
                affine,
                [0.0, 0.0, 1.0]
            ])

            return H.astype(np.float64), mask

        except Exception:

            return None, None

    # =====================================================
    # RMSE
    # =====================================================

    def _reprojection_rmse(
        self,
        H,
        source_points,
        target_points,
        mask
    ):

        if (
            H is None
            or source_points is None
            or target_points is None
            or mask is None
        ):
            return None

        keep = mask.ravel().astype(bool)

        if keep.sum() == 0:
            return None

        source_inliers = source_points[keep]
        target_inliers = target_points[keep]

        projected = cv2.perspectiveTransform(
            source_inliers,
            H
        )

        error = (
            projected.reshape(-1, 2)
            - target_inliers.reshape(-1, 2)
        )

        return float(
            np.sqrt(
                np.mean(
                    np.sum(
                        error ** 2,
                        axis=1
                    )
                )
            )
        )

    # =====================================================
    # SPATIAL COVERAGE
    # =====================================================

    def _spatial_coverage(
        self,
        keypoints,
        matches,
        mask,
        shape
    ):

        if (
            not keypoints
            or not matches
            or mask is None
        ):
            return 0.0

        keep = mask.ravel().astype(bool)

        points = []

        for match, valid in zip(
            matches,
            keep
        ):

            if valid:

                points.append(
                    keypoints[
                        match.queryIdx
                    ].pt
                )

        if len(points) < 3:
            return 0.0

        points = np.asarray(
            points,
            dtype=np.float32
        )

        height, width = shape[:2]

        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])

        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])

        box_area = max(
            0.0,
            (x_max - x_min)
            * (y_max - y_min)
        )

        image_area = max(
            1.0,
            float(width * height)
        )

        return float(
            min(
                1.0,
                box_area / image_area
            )
        )

    # =====================================================
    # HOMOGRAPHY STABILITY
    # =====================================================

    def _homography_stability(self, H):

        if H is None:
            return 0.0

        try:

            if abs(H[2, 2]) < 1e-12:
                return 0.0

            Hn = H / H[2, 2]

            if not np.all(
                np.isfinite(Hn)
            ):
                return 0.0

            determinant = abs(
                float(
                    np.linalg.det(
                        Hn[:2, :2]
                    )
                )
            )

            if determinant <= 0:
                return 0.0

            projective = (
                abs(float(Hn[2, 0]))
                + abs(float(Hn[2, 1]))
            )

            stability = (
                1.0
                / (
                    1.0
                    + projective * 100.0
                )
            )

            if (
                determinant < 0.05
                or determinant > 20.0
            ):

                stability *= 0.35

            return float(
                np.clip(
                    stability,
                    0.0,
                    1.0
                )
            )

        except Exception:

            return 0.0

    # =====================================================
    # QUALITY SCORE
    # =====================================================

    def _quality_score(
        self,
        inlier_ratio,
        coverage,
        inliers,
        rmse,
        stability
    ):

        ratio_score = np.clip(
            inlier_ratio / 0.65,
            0,
            1
        )

        coverage_score = np.clip(
            coverage / 0.35,
            0,
            1
        )

        inlier_score = np.clip(
            inliers / 80.0,
            0,
            1
        )

        if rmse is None:

            rmse_score = 0.0

        else:

            rmse_score = (
                1.0
                - np.clip(
                    rmse / 12.0,
                    0,
                    1
                )
            )

        score = (

            0.34 * ratio_score

            + 0.22 * coverage_score

            + 0.22 * inlier_score

            + 0.12 * rmse_score

            + 0.10 * stability
        )

        return float(
            np.clip(
                score * 100.0,
                0,
                100
            )
        )

    # =====================================================
    # DRAW CORRESPONDENCES
    # =====================================================

    def _draw_matches(
        self,
        source,
        kp_source,
        target,
        kp_target,
        matches,
        mask
    ):

        if not matches:

            return np.hstack([
                source,
                target
            ])

        if mask is not None:

            valid = mask.ravel().astype(bool)

            verified_matches = [
                match
                for match, keep
                in zip(matches, valid)
                if keep
            ]

        else:

            verified_matches = matches

        if not verified_matches:

            verified_matches = matches

        try:

            flags = (
                cv2.DrawMatchesFlags
                .NOT_DRAW_SINGLE_POINTS
            )

        except AttributeError:

            flags = 2

        return cv2.drawMatches(
            source,
            kp_source,
            target,
            kp_target,
            verified_matches,
            None,
            matchColor=(80, 220, 120),
            singlePointColor=(255, 180, 60),
            flags=flags
        )

    # =====================================================
    # SAVE IMAGE
    # =====================================================

    def _save_image(
        self,
        image,
        output_dir,
        filename
    ):

        path = (
            Path(output_dir)
            / filename
        )

        success = cv2.imwrite(
            str(path),
            image
        )

        if not success:
            raise ValueError(
                f"Could not save result image: {path}"
            )

        return str(path)

    # =====================================================
    # MAIN ANALYSIS
    # =====================================================

    def analyze(
        self,
        source_path,
        reference_path,
        output_dir
    ):

        start_time = time.perf_counter()

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # -------------------------------------------------
        # Load
        # -------------------------------------------------

        source_color = self._load_color(
            source_path
        )

        target_color = self._load_color(
            reference_path
        )

        source_gray = cv2.cvtColor(
            source_color,
            cv2.COLOR_BGR2GRAY
        )

        target_gray = cv2.cvtColor(
            target_color,
            cv2.COLOR_BGR2GRAY
        )

        # -------------------------------------------------
        # Preprocess
        # -------------------------------------------------

        source_processed = self._preprocess(
            source_gray
        )

        target_processed = self._preprocess(
            target_gray
        )

        # -------------------------------------------------
        # Detect
        # -------------------------------------------------

        kp_source, des_source = self._detect(
            source_processed
        )

        kp_target, des_target = self._detect(
            target_processed
        )

        # -------------------------------------------------
        # First matching pass
        # -------------------------------------------------

        mutual_matches, forward_matches = (
            self._mutual_matches(
                des_source,
                des_target,
                self.ratio_threshold
            )
        )

        matches = mutual_matches
        matching_strategy = "Mutual Lowe Ratio"

        # -------------------------------------------------
        # Relaxed pass
        # -------------------------------------------------

        if len(matches) < max(
            8,
            self.min_inliers
        ):

            relaxed_mutual, relaxed_forward = (
                self._mutual_matches(
                    des_source,
                    des_target,
                    self.fallback_ratio
                )
            )

            if len(relaxed_mutual) > len(matches):

                matches = relaxed_mutual

                matching_strategy = (
                    "Relaxed Mutual Lowe Ratio"
                )

            if len(matches) < max(
                8,
                self.min_inliers
            ):

                if len(relaxed_forward) > len(matches):

                    matches = relaxed_forward

                    matching_strategy = (
                        "Relaxed Lowe Ratio"
                    )

        # -------------------------------------------------
        # Convert matches to points
        # -------------------------------------------------

        source_points, target_points = (
            self._points_from_matches(
                kp_source,
                kp_target,
                matches
            )
        )

        # -------------------------------------------------
        # Homography
        # -------------------------------------------------

        H, mask = self._homography(
            source_points,
            target_points
        )

        model = "Homography"

        inliers = (
            int(mask.sum())
            if mask is not None
            else 0
        )

        # -------------------------------------------------
        # Affine fallback
        # -------------------------------------------------

        if (
            H is None
            or inliers < 4
        ):

            affine_H, affine_mask = (
                self._affine_fallback(
                    source_points,
                    target_points
                )
            )

            affine_inliers = (
                int(affine_mask.sum())
                if affine_mask is not None
                else 0
            )

            if (
                affine_H is not None
                and affine_inliers > inliers
            ):

                H = affine_H
                mask = affine_mask
                inliers = affine_inliers

                model = "Affine RANSAC"

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        candidate_matches = len(matches)

        inlier_ratio = (
            inliers / candidate_matches
            if candidate_matches > 0
            else 0.0
        )

        rmse = self._reprojection_rmse(
            H,
            source_points,
            target_points,
            mask
        )

        coverage = self._spatial_coverage(
            kp_source,
            matches,
            mask,
            source_gray.shape
        )

        stability = self._homography_stability(
            H
        )

        quality = self._quality_score(
            inlier_ratio,
            coverage,
            inliers,
            rmse,
            stability
        )

        # -------------------------------------------------
        # Reliability
        # -------------------------------------------------

        reliable = (

            H is not None

            and inliers >= self.min_inliers

            and inlier_ratio >= 0.15
        )

        # -------------------------------------------------
        # Correspondence visualization
        # -------------------------------------------------

        correspondence_image = (
            self._draw_matches(
                source_color,
                kp_source,
                target_color,
                kp_target,
                matches,
                mask
            )
        )

        correspondence_path = self._save_image(
            correspondence_image,
            output_dir,
            (
                f"correspondences_"
                f"{Path(source_path).stem}.jpg"
            )
        )

        # -------------------------------------------------
        # Registered + overlay
        # -------------------------------------------------

        registered_path = ""
        overlay_path = ""

        if H is not None:

            target_height, target_width = (
                target_color.shape[:2]
            )

            try:

                registered = cv2.warpPerspective(
                    source_color,
                    H,
                    (
                        target_width,
                        target_height
                    ),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )

                overlay = cv2.addWeighted(
                    target_color,
                    0.50,
                    registered,
                    0.50,
                    0
                )

                registered_path = self._save_image(
                    registered,
                    output_dir,
                    (
                        f"registered_"
                        f"{Path(source_path).stem}.jpg"
                    )
                )

                overlay_path = self._save_image(
                    overlay,
                    output_dir,
                    (
                        f"overlay_"
                        f"{Path(source_path).stem}.jpg"
                    )
                )

            except Exception as e:

                print(
                    "Registration warning:",
                    str(e)
                )

        # -------------------------------------------------
        # Processing time
        # -------------------------------------------------

        processing_time = (
            time.perf_counter()
            - start_time
        )

        # -------------------------------------------------
        # Failure reason
        # -------------------------------------------------

        failure_reason = ""

        if not reliable:

            if H is None:

                failure_reason = (
                    "A stable geometric transformation "
                    "could not be estimated from the "
                    "detected correspondences."
                )

            elif inliers < self.min_inliers:

                failure_reason = (
                    f"Only {inliers} geometrically "
                    f"consistent inliers were found; "
                    f"{self.min_inliers} are required."
                )

            else:

                failure_reason = (
                    "The geometric consistency ratio "
                    "was too low."
                )

        # =================================================
        # FINAL RESULT
        # =================================================

        result = {

            "success": bool(reliable),

            "reliable": bool(reliable),

            "status": (
                "Reliable correspondence established"
                if reliable
                else
                "No reliable correspondence established"
            ),

            "failure_reason": failure_reason,

            # Images
            "source_image": str(source_path),
            "reference_image": str(reference_path),

            "source_path": str(source_path),
            "reference_path": str(reference_path),

            "matches_image": correspondence_path,
            "correspondences_image": correspondence_path,
            "verification_image": correspondence_path,

            "registered_image": registered_path,
            "overlay_image": overlay_path,
            "mosaic_image": registered_path,

            # Keypoints
            "source_keypoints": int(
                len(kp_source)
            ),

            "reference_keypoints": int(
                len(kp_target)
            ),

            "target_keypoints": int(
                len(kp_target)
            ),

            # Matching
            "raw_matches": int(
                len(forward_matches)
            ),

            "candidate_matches": int(
                candidate_matches
            ),

            "mutual_matches": int(
                len(matches)
            ),

            # RANSAC
            "ransac_inliers": int(
                inliers
            ),

            "inliers": int(
                inliers
            ),

            "inlier_ratio": float(
                inlier_ratio * 100.0
            ),

            "inlier_ratio_fraction": float(
                inlier_ratio
            ),

            # Geometry
            "spatial_coverage": float(
                coverage * 100.0
            ),

            "reprojection_rmse": (
                float(rmse)
                if rmse is not None
                else None
            ),

            "rmse": (
                float(rmse)
                if rmse is not None
                else None
            ),

            "homography_stability": float(
                stability * 100.0
            ),

            # Quality
            "quality_score": float(
                quality
            ),

            "confidence": float(
                quality
            ),

            # Technical information
            "model": model,

            "geometric_model": model,

            "matching_strategy":
                matching_strategy,

            "detector":
                self.detector_name,

            "ratio_threshold":
                float(self.ratio_threshold),

            "ransac_threshold":
                float(self.ransac_threshold),

            "min_inliers":
                int(self.min_inliers),

            "processing_time":
                float(processing_time),

            "processing_time_seconds":
                float(processing_time)
        }

        return result


# =========================================================
# BACKWARD COMPATIBILITY
# =========================================================

LunarAnalysisEngine = LunarCorrespondenceEngine


# =========================================================
# HELPER FUNCTION
# =========================================================

def run_analysis(
    source_path,
    reference_path,
    output_dir,
    detector="SIFT",
    ratio_threshold=0.75,
    ransac_threshold=5.0,
    min_inliers=10
):

    engine = LunarCorrespondenceEngine(

        detector=detector,

        ratio_threshold=ratio_threshold,

        ransac_threshold=ransac_threshold,

        min_inliers=min_inliers
    )

    return engine.analyze(

        source_path,

        reference_path,

        output_dir
    )