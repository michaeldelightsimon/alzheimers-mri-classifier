import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
from matplotlib import colormaps
from torchvision import models, transforms

# ImageFolder sorts class folders alphabetically, so this order matches training
CLASS_NAMES = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
MODEL_PATH = "alz_resnet50.pth"

MODEL_TEST_ACCURACY = 88  # % from your Colab test-set report; update if you retrain

REPORT = {
    "NonDemented": {
        "title": "No signs of dementia detected",
        "level": "success",
        "meaning": (
            "The model did not find patterns in this scan that it associates with dementia. "
            "This does not rule out memory problems, because a single image cannot show everything."
        ),
        "next": [
            "No urgent action is suggested from this result.",
            "If you or a family member notice memory problems, mention them to a doctor anyway.",
            "Healthy habits help the brain: regular exercise, good sleep, social activity, and keeping blood pressure under control.",
        ],
    },
    "VeryMildDemented": {
        "title": "Very mild (early) changes",
        "level": "info",
        "meaning": (
            "The patterns here resemble the earliest stage. People at this stage often have small memory "
            "lapses, such as misplacing items or forgetting recent conversations, and usually manage daily "
            "life on their own. Some early changes also appear in normal ageing."
        ),
        "next": [
            "Book a routine check-up with a doctor, ideally a neurologist or memory clinic.",
            "Simple memory tests and blood tests usually follow to find the cause.",
            "Early review is useful because lifestyle changes and treatment can help slow some symptoms.",
        ],
    },
    "MildDemented": {
        "title": "Mild dementia pattern",
        "level": "warning",
        "meaning": (
            "The patterns here resemble mild dementia. Memory loss starts to affect daily life: trouble finding "
            "words, planning, or handling money, and sometimes getting lost in familiar places. Many people at "
            "this stage can still do most things independently, with some support."
        ),
        "next": [
            "See a doctor soon for a full assessment (memory tests, blood tests, specialist review).",
            "Ask about treatment options and support services.",
            "Involve family early to plan support and home safety.",
        ],
    },
    "ModerateDemented": {
        "title": "Moderate dementia pattern: medical review recommended promptly",
        "level": "error",
        "meaning": (
            "The patterns here resemble moderate dementia. This stage usually brings more confusion, difficulty "
            "with daily tasks such as dressing or cooking, changes in behaviour or sleep, and a need for regular "
            "help from other people."
        ),
        "next": [
            "See a doctor or neurologist promptly for assessment and a care plan.",
            "Arrange day-to-day support and safety measures at home, together with the family.",
            "Seek urgent medical help for sudden worsening, severe confusion, or if the person wanders and cannot be found.",
        ],
    },
}

preprocess = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@st.cache_resource
def load_model():
    t0 = time.perf_counter()
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, len(CLASS_NAMES)))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, 224, 224))  # warm-up so the first real image is not slower
    return model, time.perf_counter() - t0


def grad_cam(model, x, class_idx):
    """Grad-CAM: shows which image regions pushed the model toward class_idx."""
    store = {}

    def forward_hook(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))

    handle = model.layer4[-1].register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            model.zero_grad()
            out = model(x)
            out[0, class_idx].backward()
    finally:
        handle.remove()

    act = store["act"][0].detach()          # (C, h, w)
    grad = store["grad"][0].detach()        # (C, h, w)
    weights = grad.mean(dim=(1, 2))         # one weight per channel
    cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
    cam = cam / (cam.max() + 1e-8)
    return cam.numpy()


def make_heatmap(img_gray, cam):
    """Return (original, heatmap) as uint8 RGB arrays of size 224x224. The original is left untouched."""
    base = np.asarray(img_gray.resize((224, 224)).convert("RGB"), dtype=np.uint8)
    cam_t = torch.from_numpy(cam)[None, None]
    cam_up = F.interpolate(cam_t, size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
    cam_up = np.clip(cam_up, 0, 1)
    heat = (colormaps["jet"](cam_up)[..., :3] * 255).astype(np.uint8)
    return base, heat


st.set_page_config(page_title="Alzheimer's MRI Classifier", page_icon="🧠", layout="wide")
st.title("🧠 Alzheimer's Stage Classifier")
st.caption("Student project. Not a medical diagnosis tool.")

model, load_time = load_model()
st.sidebar.metric("Model load time", f"{load_time:.2f} s")
st.sidebar.write("Running on: **CPU**")
show_cam = st.sidebar.checkbox("Show heatmap (Grad-CAM)", value=True)
st.sidebar.caption("Confidence is how sure the model is about this one image. Test accuracy is how often it was right on images it had never seen.")

file = st.file_uploader("Upload a brain MRI slice", type=["png", "jpg", "jpeg"])

if file:
    raw = Image.open(file)
    rgb = np.asarray(raw.convert("RGB"), dtype=np.int16)
    color_spread = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
    img = raw.convert("L")
    if color_spread > 6:
        st.warning(
            "This image has colour in it (for example a heatmap, overlay or edited picture). The model was trained "
            "only on plain grayscale MRI slices, so the result below is unreliable. Upload the original MRI image instead."
        )

    t0 = time.perf_counter()
    x = preprocess(img).unsqueeze(0)
    t1 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    t2 = time.perf_counter()

    top = int(probs.argmax())
    name = CLASS_NAMES[top]
    rep = REPORT[name]
    conf = probs[top].item()

    st.subheader(f"Prediction: {name}")
    m1, m2 = st.columns(2)
    m1.metric("Confidence for this image", f"{conf * 100:.1f}%")
    m2.metric("Model overall test accuracy", f"{MODEL_TEST_ACCURACY}%")

    getattr(st, rep["level"])(f"**{rep['title']}**")

    top2 = torch.topk(probs, 2)
    second_name = CLASS_NAMES[int(top2.indices[1])]
    second_conf = top2.values[1].item()
    if conf < 0.60:
        st.warning(
            f"Low confidence: the result may be wrong. The model's second choice is "
            f"**{second_name}** ({second_conf * 100:.1f}%). Treat this as unreliable."
        )
    elif conf - second_conf < 0.20:
        st.info(f"Close call: the model also considered **{second_name}** ({second_conf * 100:.1f}%).")

    st.markdown("#### What this means")
    st.write(rep["meaning"])
    st.markdown("#### Suggested next steps")
    for step in rep["next"]:
        st.markdown(f"- {step}")
    st.caption(
        "This is an AI demo built for a class project. It looks at a single MRI image and cannot diagnose anything. "
        "Only a doctor can confirm a diagnosis."
    )

    cam_time = None
    if show_cam:
        t3 = time.perf_counter()
        cam = grad_cam(model, x, top)
        original, heat = make_heatmap(img, cam)
        cam_time = time.perf_counter() - t3

        c1, c2 = st.columns(2)
        c1.image(original, caption="Original MRI", width=320)
        c2.image(heat, caption=f"Heatmap: areas that drove '{CLASS_NAMES[top]}'", width=320)
        st.caption(
            "Red and yellow mark the areas that influenced the model's prediction most; blue areas had little "
            "influence. This shows where the model looked, not a confirmed location of disease."
        )

    st.bar_chart(pd.DataFrame({"probability": probs.numpy()}, index=CLASS_NAMES))

    cols = st.columns(4 if cam_time is not None else 3)
    cols[0].metric("Preprocessing", f"{(t1 - t0) * 1000:.0f} ms")
    cols[1].metric("Model inference", f"{(t2 - t1) * 1000:.0f} ms")
    cols[2].metric("Total (prediction)", f"{(t2 - t0) * 1000:.0f} ms")
    if cam_time is not None:
        cols[3].metric("Heatmap", f"{cam_time * 1000:.0f} ms")