from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import torch

app = Flask(__name__)

# Load the Nepali sentence similarity model
model_name = "l3cube-pune/indic-sentence-similarity-sbert"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load your array of sentences
sentence_array = [
    "घर/जग्गा नामसारीको सिफारिस गरी पाऊँ",
    "मोही लगत कट्टाको सिफारिस पाउं",
    "घर कायम सिफारीस पाउं",
    "अशक्त सिफारिस",
    "छात्रबृत्तिको लागि सिफारिस पाऊँ",
    "आदिवासी जनजाति प्रमाणित गरी पाऊँ",
    "अस्थायी बसोबासको सिफारिस पाऊँ",
    "स्थायी बसोबासको सिफारिस गरी पाऊँ",
    "आर्थिक अवस्था कमजोर सिफारिस पाऊँ",
    "नयाँ घरमा विद्युत जडान सिफारिस पाऊँ",
    "धारा जडान सिफारिस पाऊँ",
    "दुबै नाम गरेको ब्यक्ति एक्कै हो भन्ने सिफारिस पाऊँ",
    "ब्यवसाय बन्द सिफारिस पाऊँ",
    "व्यवसाय ठाउँसारी सिफारिस पाऊँ",
    "कोर्ट–फिमिनाहा सिफारिस पाऊँ",
    "नाबालक सिफारिस पाऊँ",
    "चौपाया सिफारिस पाऊँ",
    "संस्था दर्ता गरी पाऊँ",
    "विद्यालय ठाउँसारी सिफारिस पाऊँ",
    "विद्यालय संचालन/कक्षा बृद्धिको सिफारिस पाऊँ",
    "जग्गा दर्ता सिफारिस पाऊँ",
    "संरक्षक सिफारिस पाऊँ",
    "बाटो कायम सिफारिस पाऊँ",
    "जिवित नाता प्रमाणित गरी पाऊँ",
    "मृत्यु नाता प्रमाणित गरी पाऊँ",
    "निःशुल्क स्वास्थ्य उपचारको लागि सिफारिस पाऊँ",
    "संस्था दर्ता सिफारिस पाऊँ",
    "घर बाटो प्रमाणित गरी पाऊँ",
    "चारकिल्ला प्रमाणित गरि पाउ",
    "जन्म मिति प्रमाणित गरि पाउ",
    "बिवाह प्रमाणित गरि पाऊँ",
    "घर पाताल प्रमाणित गरी पाऊँ",
    "हकदार प्रमाणित गरी पाऊँ",
    "अबिवाहित प्रमाणित गरी पाऊँ",
    "जग्गाधनी प्रमाण पूर्जा हराएको सिफारिस पाऊँ",
    "व्यवसाय दर्ता गरी पाऊँ",
    "मोही नामसारीको लागि सिफारिस गरी पाऊँ",
    "मूल्याङ्कन गरी पाऊँ",
    "तीन पुस्ते खोली सिफारिस गरी पाऊँ",
    "पुरानो घरमा विद्युत जडान सिफारिस पाऊँ",
    "सामाजिक सुरक्षा भत्ता नाम दर्ता सम्बन्धमा",
    "बहाल समझौता",
    "कोठा खोली पाऊँ",
    "अपाङ्ग सिफारिस पाऊँ",
    "नापी नक्सामा बाटो नभएको फिल्डमा बाटो भएको सिफारिस",
    "धारा नामसारी सिफारिस पाऊँ",
    "विद्युत मिटर नामसारी सिफारिस",
    "फोटो टाँसको लागि तीन पुस्ते खोली सिफारिस पाऊ",
    "कोठा बन्द सिफारिस पाऊँ",
    "अस्थाई टहराको सम्पत्ति कर तिर्न सिफारिस गरी पाऊँ",
    "औषधि उपचार बापत खर्च पाउँ भन्ने सम्वन्धमा",
    "नागरिकता र प्रतिलिपि सिफारिस",
    "अंग्रेजीमा सिफारिस"
]

@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    try:
        #get data from form data
        voice_text = request.form['voice_text']

        # Sample voice input (replace this with your actual voice-to-text conversion)
        """ voice_text = "कस्तो सिफारिस गर्नुपर्छ अस्थायी बसोबासको?" """

        # Tokenize the voice_text
        voice_inputs = tokenizer(voice_text, return_tensors="pt")

        # Get the model output (embedding)
        model_output = model(**voice_inputs).pooler_output

        # Calculate similarity scores
        similarity_scores = []
        for sentence in sentence_array:
            sentence_inputs = tokenizer(sentence, return_tensors="pt")
            sentence_output = model(**sentence_inputs).pooler_output

            # You might want to use a different score if the model provides one
            similarity_score = torch.nn.functional.cosine_similarity(model_output, sentence_output).item()
            similarity_scores.append(similarity_score)

        # Find the most similar sentence
        max_similarity_index = similarity_scores.index(max(similarity_scores))
        most_similar_sentence = sentence_array[max_similarity_index]

        result = {
            "voice_text": voice_text,
            "most_similar_sentence": most_similar_sentence,
            "max_similarity_index" : max_similarity_index,
            "similarity_score": max(similarity_scores)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000)
