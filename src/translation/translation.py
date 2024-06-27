from transformers import MarianMTModel, MarianTokenizer

class TranslationModel:
    def __init__(self, model_name='Helsinki-NLP/opus-mt-en-de'):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        translated = self.model.generate(**inputs)
        translated_text = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        return translated_text[0]
