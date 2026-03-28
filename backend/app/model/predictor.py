import io
import os
import numpy as np
from PIL import Image
import torch
from app.model.model_loader import model
from app.utils.preprocessing import preprocess_image
from app.utils.video_utils import extract_frames
from app.utils.explainability import Explainer

# ── Global explainer instance (reuses model) ──
explainer = Explainer(model)


async def predict_file(file):
    filename = file.filename.lower()

    if filename.endswith(("jpg", "png", "jpeg")):
        return await predict_image(file)

    elif filename.endswith(("mp4", "avi")):
        return await predict_video(file)

    return {"error": "Unsupported format"}


async def predict_image(file):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = preprocess_image(image)

    # Ensure model is in eval mode
    model.eval()
    with torch.no_grad():
        prob = model(input_tensor).item()

    # ── Generate per-image specific explanation using Grad-CAM + landmarks ──
    try:
        explanation = explainer.generate_explanation(image, input_tensor, prob)
    except Exception as e:
        print(f"Explainability error: {e}")
        explanation = {
            "summary": "Deepfake probability: {:.1f}%".format(prob * 100),
            "details": ["Analysis completed. Confidence: {:.1f}%".format(prob * 100)],
            "heatmap": None
        }

    return {
        "type": "image",
        "prediction": "Fake" if prob > 0.5 else "Real",
        "confidence": prob,
        "reason": explanation
    }


async def predict_video(file):
    import requests
    path = f"temp_{file.filename}"

    with open(path, "wb") as f:
        f.write(await file.read())

    # ── 1. SIGHTENGINE API CALL FOR VERDICT ──
    sightengine_prob = None
    frames = extract_frames(path)
    os.remove(path)
    
    if len(frames) > 0:
        try:
            # Save the first extracted frame to a temp file for Sightengine
            temp_img_path = f"temp_frame_{file.filename}.jpg"
            img_to_send = Image.fromarray(frames[0])
            img_to_send.save(temp_img_path, format="JPEG")

            url = "https://api.sightengine.com/1.0/check.json"
            params = {
                "models": "deepfake",
                "api_user": "256580568",
                "api_secret": "83CfxEs7hMtYbAm8QiNxk3sjtxoyufpR"
            }
            with open(temp_img_path, "rb") as frame_file:
                se_resp = requests.post(url, files={"media": frame_file}, data=params, timeout=15)
            
            data = se_resp.json()
            print("================ SIGHTENGINE RAW RESPONSE ================\n", data)
            
            if data.get("status") == "success" and "type" in data:
                sightengine_prob = float(data["type"].get("deepfake", 0.0))
                print(f"✅ Sightengine Confidence Extracted: {sightengine_prob*100:.2f}%")
            
            # Clean up temp image
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
                
        except Exception as e:
            print(f"⚠️ Sightengine API Exception: {e}")

    # ── 2. LOCAL XAI PIPELINE FOR HEATMAPS ──

    probs = []
    explanations = []
    heatmap_url = None

    model.eval()

    for i, frame in enumerate(frames):
        image = Image.fromarray(frame)
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        image = Image.open(buffer)

        tensor = preprocess_image(image)

        with torch.no_grad():
            prob = model(tensor).item()
            probs.append(prob)

        # Generate explanation using local prob or sightengine_prob if available
        xai_prob = sightengine_prob if sightengine_prob is not None else prob
        if i == 0 or xai_prob > 0.7:
            try:
                expl = explainer.generate_explanation(image, tensor, xai_prob)
                explanations.append(expl)
                if expl.get("heatmap") and (heatmap_url is None or xai_prob > 0.7):
                    heatmap_url = expl["heatmap"]
            except Exception as e:
                print(f"Video frame explainability error: {e}")

    # Use Sightengine's precise score if available; otherwise use our local CNN average
    avg = sightengine_prob if sightengine_prob is not None else float(np.mean(probs))

    if explanations:
        best_expl = max(explanations, key=lambda x: len(x.get("details", [])))
        summary = best_expl.get("summary", "")
        details = best_expl.get("details", [])
    else:
        summary = "Video analysis complete."
        details = [f"Analyzed {len(probs)} frames. Confidence: {avg*100:.1f}%"]

    fake_frames = sum(1 for p in probs if p > 0.5)
    details.append(f"Local AI flagged {fake_frames}/{len(probs)} individual frames as suspicious.")
    if sightengine_prob is not None:
        details.insert(0, "✅ Verified by Sightengine Enterprise AI.")
    
    if len(probs) > 1:
        details.append(f"Confidence range across frames: {min(probs)*100:.1f}% - {max(probs)*100:.1f}%.")

    return {
        "type": "video",
        "prediction": "Fake" if avg > 0.5 else "Real",
        "confidence": avg,
        "frames": len(probs),
        "reason": {
            "summary": summary,
            "details": details,
            "heatmap": heatmap_url
        }
    }