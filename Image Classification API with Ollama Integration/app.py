from flask import Flask, request, jsonify, render_template
from ai import list_models, describe_image

app = Flask(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/models", methods=["GET"])
def models():
    """Return list of available Ollama models (names only)."""
    return jsonify({"success": True, "models": list_models()})


@app.route("/classify", methods=["POST"])
def classify():
    """
    MULTI-IMAGE AI description / focus endpoint.

    Accepts (multipart/form-data):
      - image: multiple files allowed
      - model: optional model name
      - focus: optional object name (describe only this object)
    """
    files = request.files.getlist("image")
    if not files:
        return jsonify({"success": False, "error": "No image files provided"}), 400

    # Validate allowed formats
    valid_files = [f for f in files if allowed_file(f.filename)]
    if not valid_files:
        return jsonify({"success": False, "error": "Unsupported or missing image files"}), 400

    model = request.form.get("model") or None
    focus = request.form.get("focus") or None

    # Auto-pick first model if not specified
    if not model:
        models_list = list_models()
        if not models_list:
            return jsonify({"success": False, "error": "No models available from Ollama"}), 500
        model = models_list[0]

    # Process each image
    results = []
    for image in valid_files:
        try:
            res = describe_image(model=model, image_file=image, focus=focus)
            results.append(res)
        except Exception as e:
            results.append({
                "success": False,
                "model": model,
                "error": str(e)
            })

    return jsonify({"success": True, "results": results}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
