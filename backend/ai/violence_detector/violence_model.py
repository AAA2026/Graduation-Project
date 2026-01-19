# Placeholder for actual violence detection model integration

class ViolenceModel:
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def predict(self, video_path: str) -> float:
        """
        Returns a probability (0.0 - 1.0) of violence in the video.
        Replace with real inference using loaded model.
        """
        raise NotImplementedError("ViolenceModel.predict is not implemented yet")
