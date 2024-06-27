import cv2
import pytesseract

class ImageRecognitionModel:
    def recognize_text(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Image at path '{image_path}' could not be loaded.")
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            return f"Error recognizing text: {str(e)}"
