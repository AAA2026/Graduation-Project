import gradio as gr
import torch
import os

from inference import load_model, predict_video

# ============================================================
#                CONFIGURATION
# ============================================================
MODEL_PATH = r"D:\CarCrash_Final_Model\mobilenetv2_lstm_finetuned.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔥 Using device: {DEVICE}")
print(f"🔥 Loading model from: {MODEL_PATH}")

# Load model once at startup
model = load_model(MODEL_PATH, device=DEVICE)
print("✅ Model loaded successfully!")


# ============================================================
#                GRADIO PREDICTION FUNCTION
# ============================================================
def gradio_predict(video_file):
    if video_file is None:
        return "❌ No video uploaded."

    try:
        result = predict_video(
            model,
            video_file,
            num_frames=16,
            img_size=224,
            device=DEVICE
        )

        normal_p = result["normal"]
        accident_p = result["accident"]
        pred = result["prediction"].upper()

        out_text = (
            f"### 🔍 Prediction Result\n"
            f"- **Normal:** {normal_p:.4f}\n"
            f"- **Accident:** {accident_p:.4f}\n\n"
            f"## 🚨 Final Prediction: **{pred}**"
        )
        return out_text

    except Exception as e:
        return f"❌ Error while processing video:\n```\n{str(e)}\n```"


# ============================================================
#                GRADIO APP UI DESIGN
# ============================================================
css = """
.gradio-container {
    background-color: #111 !important;
    color: #eee !important;
}
h1 {
    text-align: center;
    color: #00aced !important;
}
h2 {
    color: #bbb !important;
}
"""

with gr.Blocks(css=css, title="Accident Detection System") as demo:

    gr.Markdown("# 🚗 Accident Detection System (MobileNetV2 + LSTM)")
    gr.Markdown("### Upload a road video and let the AI detect **NORMAL** or **ACCIDENT**.")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="📤 Upload Road Video",
                sources=["upload"],
            )
            analyze_btn = gr.Button("🔍 Analyze Video", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Markdown("### Prediction will appear here...")

    analyze_btn.click(fn=gradio_predict, inputs=video_input, outputs=output_box)

demo.launch(server_name="0.0.0.0", server_port=7860)
