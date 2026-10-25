from flask import Flask, render_template, request, send_from_directory
from pathlib import Path
import uuid

from services.analysis_engine import LunarCorrespondenceEngine


# =========================================================
# APP SETUP
# =========================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULT_DIR = BASE_DIR / "data" / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# DEFAULT SETTINGS
# =========================================================

DEFAULT_DETECTOR = "SIFT"
DEFAULT_MIN_INLIERS = 10
DEFAULT_RATIO = 0.75
DEFAULT_RANSAC = 5.0


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# =========================================================
# ANALYZE
# =========================================================

@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    if request.method == "GET":

        return render_template(
            "analyze.html",
            default_detector=DEFAULT_DETECTOR,
            default_ratio=DEFAULT_RATIO,
            default_ransac=DEFAULT_RANSAC
        )

    # -----------------------------------------------------
    # POST
    # -----------------------------------------------------

    source_file = request.files.get("source_image")
    reference_file = request.files.get("reference_image")

    # -----------------------------------------------------
    # Validate source image
    # -----------------------------------------------------

    if not source_file or not source_file.filename:

        return render_template(
            "analyze.html",
            error="Please select a source image.",
            default_detector=DEFAULT_DETECTOR,
            default_ratio=DEFAULT_RATIO,
            default_ransac=DEFAULT_RANSAC
        )

    # -----------------------------------------------------
    # Validate reference image
    # -----------------------------------------------------

    if not reference_file or not reference_file.filename:

        return render_template(
            "analyze.html",
            error="Please select a reference image.",
            default_detector=DEFAULT_DETECTOR,
            default_ratio=DEFAULT_RATIO,
            default_ransac=DEFAULT_RANSAC
        )

    # =====================================================
    # READ SETTINGS
    # =====================================================

    detector = request.form.get(
        "detector",
        DEFAULT_DETECTOR
    ).upper()

    try:

        ratio_threshold = float(
            request.form.get(
                "ratio_threshold",
                DEFAULT_RATIO
            )
        )

    except (ValueError, TypeError):

        ratio_threshold = DEFAULT_RATIO

    try:

        ransac_threshold = float(
            request.form.get(
                "ransac_threshold",
                DEFAULT_RANSAC
            )
        )

    except (ValueError, TypeError):

        ransac_threshold = DEFAULT_RANSAC

    # -----------------------------------------------------
    # Keep values within safe range
    # -----------------------------------------------------

    ratio_threshold = max(
        0.50,
        min(ratio_threshold, 0.95)
    )

    ransac_threshold = max(
        1.0,
        min(ransac_threshold, 20.0)
    )

    # =====================================================
    # CREATE UNIQUE ANALYSIS ID
    # =====================================================

    analysis_id = uuid.uuid4().hex[:12]

    source_ext = (
        Path(source_file.filename).suffix.lower()
        or ".png"
    )

    reference_ext = (
        Path(reference_file.filename).suffix.lower()
        or ".png"
    )

    source_path = (
        UPLOAD_DIR /
        f"{analysis_id}_source{source_ext}"
    )

    reference_path = (
        UPLOAD_DIR /
        f"{analysis_id}_reference{reference_ext}"
    )

    # =====================================================
    # SAVE IMAGES
    # =====================================================

    source_file.save(str(source_path))
    reference_file.save(str(reference_path))

    print("\n========================================")
    print("LUNARVISION ANALYSIS")
    print("========================================")
    print("Source    :", source_path.name)
    print("Reference :", reference_path.name)
    print("Detector  :", detector)
    print("Ratio     :", ratio_threshold)
    print("RANSAC    :", ransac_threshold)
    print("========================================\n")

    # =====================================================
    # RUN CORRESPONDENCE ENGINE
    # =====================================================

    try:

        engine = LunarCorrespondenceEngine(

            detector=detector,

            ratio_threshold=ratio_threshold,

            ransac_threshold=ransac_threshold,

            min_inliers=DEFAULT_MIN_INLIERS
        )

        result = engine.analyze(

            str(source_path),

            str(reference_path),

            str(RESULT_DIR)
        )

    except Exception as e:

        print("\n========================================")
        print("LUNAR ANALYSIS ERROR")
        print("========================================")
        print(type(e).__name__, ":", str(e))
        print("========================================\n")

        return render_template(

            "analyze.html",

            error=f"Analysis failed: {str(e)}",

            default_detector=detector,

            default_ratio=ratio_threshold,

            default_ransac=ransac_threshold

        ), 500

    # =====================================================
    # CHECK RESULT
    # =====================================================

    if result is None:

        return render_template(

            "analyze.html",

            error="Analysis returned no result.",

            default_detector=detector,

            default_ratio=ratio_threshold,

            default_ransac=ransac_threshold

        ), 500

    # -----------------------------------------------------
    # Make sure result is dictionary
    # -----------------------------------------------------

    if not isinstance(result, dict):

        result = {

            "success": False,

            "error":
                "Invalid analysis result returned by engine."

        }

    # =====================================================
    # RESULT URL NORMALIZATION
    # =====================================================

    def make_result_url(value):

        if not value:

            return ""

        value = str(value).replace("\\", "/")

        # Already a web URL
        if value.startswith("/"):

            return value

        # Absolute filesystem path
        try:

            value_path = Path(value)

            if value_path.exists():

                try:

                    relative = value_path.relative_to(
                        RESULT_DIR
                    )

                    return (
                        "/results/" +
                        relative.as_posix()
                    )

                except ValueError:

                    pass

        except Exception:

            pass

        # Remove common prefixes

        value = value.replace(
            "data/results/",
            ""
        )

        value = value.replace(
            "data\\results\\",
            ""
        )

        value = value.replace(
            "results/",
            ""
        )

        value = value.replace(
            "results\\",
            ""
        )

        return "/results/" + value.lstrip("/")

    # =====================================================
    # NORMALIZE IMAGE FIELDS
    # =====================================================

    image_fields = [

        "source_image",

        "reference_image",

        "source_path",

        "reference_path",

        "matches_image",

        "correspondences_image",

        "registered_image",

        "overlay_image",

        "verification_image",

        "mosaic_image"

    ]

    for field in image_fields:

        if field in result and result[field]:

            result[field] = make_result_url(
                result[field]
            )

    # =====================================================
    # TEMPLATE COMPATIBILITY ALIASES
    # =====================================================

    if not result.get("source_image"):

        result["source_image"] = result.get(
            "source_path",
            ""
        )

    if not result.get("reference_image"):

        result["reference_image"] = result.get(
            "reference_path",
            ""
        )

    if not result.get("correspondences_image"):

        result["correspondences_image"] = result.get(
            "matches_image",
            ""
        )

    if not result.get("matches_image"):

        result["matches_image"] = result.get(
            "correspondences_image",
            ""
        )

    # =====================================================
    # PRINT RESULT SUMMARY
    # =====================================================

    print("\n========================================")
    print("ANALYSIS COMPLETED")
    print("========================================")

    print(
        "Source Keypoints :",
        result.get("source_keypoints", "N/A")
    )

    print(
        "Reference Keypoints :",
        result.get("reference_keypoints", "N/A")
    )

    print(
        "Candidate Matches :",
        result.get("candidate_matches", "N/A")
    )

    print(
        "RANSAC Inliers :",
        result.get("ransac_inliers", "N/A")
    )

    print(
        "Inlier Ratio :",
        result.get("inlier_ratio", "N/A")
    )

    print("========================================\n")

    # =====================================================
    # RENDER RESULT PAGE
    # =====================================================

    return render_template(

        "analysis_result.html",

        result=result,

        detector=detector,

        ratio_threshold=ratio_threshold,

        ransac_threshold=ransac_threshold,

        source_image=result.get(
            "source_image",
            ""
        ),

        reference_image=result.get(
            "reference_image",
            ""
        ),

        correspondences_image=result.get(
            "correspondences_image",
            ""
        ),

        registered_image=result.get(
            "registered_image",
            ""
        ),

        overlay_image=result.get(
            "overlay_image",
            ""
        ),

        matches_image=result.get(
            "matches_image",
            ""
        ),

        verification_image=result.get(
            "verification_image",
            ""
        ),

        mosaic_image=result.get(
            "mosaic_image",
            ""
        )
    )


# =========================================================
# ENDPOINT ALIASES
# =========================================================
#
# Different existing templates may refer to the Analyze
# page using different endpoint names.
#
# All of these point to the SAME analyze() function.
# =========================================================

app.add_url_rule(

    "/analyze",

    endpoint="analyze_page",

    view_func=analyze,

    methods=["GET", "POST"]
)


app.add_url_rule(

    "/analyze",

    endpoint="run_analysis",

    view_func=analyze,

    methods=["GET", "POST"]
)


# =========================================================
# OBSERVATIONS
# =========================================================

@app.route("/observations")
def observations():

    return render_template(
        "observations.html"
    )


app.add_url_rule(

    "/observations",

    endpoint="observations_page",

    view_func=observations
)


# =========================================================
# LUNAR MAP
# =========================================================

@app.route("/lunar-map")
def lunar_map():

    return render_template(
        "lunar_map.html"
    )


app.add_url_rule(

    "/lunar-map",

    endpoint="lunar_map_page",

    view_func=lunar_map
)


# =========================================================
# RESULT FILES
# =========================================================

@app.route("/results/<path:filename>")
def result_file(filename):

    return send_from_directory(

        str(RESULT_DIR),

        filename
    )


# =========================================================
# UPLOAD FILES
# =========================================================

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):

    return send_from_directory(

        str(UPLOAD_DIR),

        filename
    )


# =========================================================
# 404
# =========================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "404.html"
    ), 404


# =========================================================
# 500
# =========================================================

@app.errorhandler(500)
def internal_server_error(error):

    return render_template(
        "500.html"
    ), 500


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    print("\n========================================")
    print("        LUNARVISION STARTED")
    print("========================================")
    print("URL: http://127.0.0.1:5000")
    print("========================================\n")

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True
    )