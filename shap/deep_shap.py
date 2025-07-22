import torch, shap, numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# ---------- 0. User config ----------
MODEL_NAME  = "clip-vit-base-patch32"
TEXT_PROMPT = "text promt"
IMG_PATH    = "./example.jpg"

# ---------- 1. Environment ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on {device}")


clip       = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
processor  = CLIPProcessor.from_pretrained(MODEL_NAME)

# ---------- 2. Wrap vision path ----------
class CLIPVisionWrapper(torch.nn.Module):
    def __init__(self, clip_model, processor, prompt, tgt_idx, dev):
        super().__init__()
        self.clip, self.tgt = clip_model, tgt_idx
        txt = processor(text=[prompt], return_tensors="pt")
        self.txt_inputs = {k: v.to(dev) for k, v in txt.items()}

    def forward(self, pixel_values):          # (N,3,224,224)
        logits = self.clip(pixel_values=pixel_values,
                        **self.txt_inputs).logits_per_image
        probs  = torch.softmax(logits, dim=-1)
        return probs[:, self.tgt].unsqueeze(1)   # (N,1)

wrapped = CLIPVisionWrapper(clip, processor, TEXT_PROMPT, 0, device)

# ---------- 3. Baseline (all-black) ----------
baseline = processor(images=[Image.new("RGB", (224, 224), 0)],
                        return_tensors="pt")["pixel_values"].to(device)

# ---------- 4. DeepExplainer ----------
explainer = shap.DeepExplainer(wrapped, baseline)

# ---------- 5. Prepare test image ----------
img_pil = Image.open(IMG_PATH).convert("RGB").resize((224, 224))
img_px  = processor(images=[img_pil],
                    return_tensors="pt")["pixel_values"].to(device)

# ---------- 6. Compute SHAP ----------
print("[INFO] Calculating SHAP values …")
shap_vals_ch_first = explainer.shap_values(img_px)[0]    # (1,3,224,224)

# ---------- 7. Convert to channels-last ----------
shap_vals = [np.transpose(shap_vals_ch_first, (0, 2, 3, 1))]  # (1,224,224,3)
img_batch = (np.array(img_pil) / 255.0)[None, ...]            # (1,224,224,3)

# ---------- 8. Visualise ----------
shap.image_plot(shap_vals, img_batch)





