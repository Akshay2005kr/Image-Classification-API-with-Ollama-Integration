import base64
import requests

OLLAMA_URL = "http://localhost:11434"

VISION_KEYWORDS = ["llava", "vision", "bakllava", "moondream", "minicpm"]


def is_vision_model(model_name: str) -> bool:
    name = (model_name or "").lower()
    return any(key in name for key in VISION_KEYWORDS)


def list_models():
    try:
        res = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        res.raise_for_status()
        data = res.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return ["llava:latest"]


def describe_image(model: str, image_file, focus: str | None = None):
    try:
        if not is_vision_model(model):
            return {
                "success": False,
                "error": f"The selected model '{model}' does not appear to support images. "
                         f"Please choose a vision model like 'llava' or 'vision'."
            }

        image_file.seek(0)
        img_bytes = image_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        focus = (focus or "").strip()


        if focus:
            system_prompt = (
                f"You are a precise image analysis assistant. The user is interested ONLY in the object "
                f"'{focus}'. Look at the entire image but describe exclusively that object. "
                "If the object is not visible, respond with exactly: 'Not visible'. "
                "Otherwise, describe its appearance, color, shape, size, pose, position in the frame, "
                "surroundings and any notable details. "
                "Write at least 10–15 full sentences in natural English."
            )
            user_prompt = "Describe ONLY the requested object in the attached image."
        else:
            system_prompt = (
                "You are a highly detailed image description assistant. Carefully observe the image and "
                "describe EVERYTHING you see without guessing. Mention the following clearly: "
                "the main subjects (animals, people, objects) and how many of each, their appearance, colors, "
                "patterns, poses, facial expressions, emotions, background, environment, textures, lighting, "
                "shadows, and relationships between elements. Include any visible symbols or readable text. "
                "Write a long multi-paragraph description with at least 18–25 sentences. "
                "If something is unclear, say it is unclear instead of inventing details."
            )
            user_prompt = "Analyze the attached image and describe it in full detail."

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt, "images": [img_b64]},
            ],
            "stream": False,
        }

        res = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=200)
        res.raise_for_status()
        data = res.json()

        message = data.get("message", {}) or {}
        description = (message.get("content") or "").strip()

        if not description:
            description = "No description returned by the model."

        return {"success": True, "model": model, "focus": focus or None, "description": description}

    except Exception as e:
        return {"success": False, "model": model, "error": str(e)}
