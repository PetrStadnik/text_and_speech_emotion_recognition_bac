import numpy as np
from funasr import AutoModel
from transformers import pipeline
from pydub import AudioSegment


class ProcessChain:
    def __init__(self):
        self._speech_model = AutoModel(model="iic/emotion2vec_base_finetuned", model_revision="v2.0.4")
        self._text_model = pipeline("text-classification", model="rahulmallah/autotrain-emotion-detection-1366352626")

        self._speech_model_dictionary = {
            0: "angry",
            3: "happy",
            6: "sad",
            8: "unknown",
            2: "fearful",
            7: "surprised",
            1: "disgusted",
            4: "neutral",
            5: "other",
        }

        self._text_model_dictionary = {
            'anger': 'angry',
            'boredom': 'bored',
            'empty': 'empty',
            'enthusiasm': 'enthusiastic',
            'fun': 'funny',
            'happiness': 'happy',
            'hate': 'hate',
            'love': 'love',
            'neutral': 'neutral',
            'relief': 'relief',
            'sadness': 'sad',
            'surprise': 'surprised',
            'worry': 'worried'
        }

        self._sentiment_dictionary = {
            "positive": ["happy", "amusement", "excited", "joy", "love", "desire", "optimism", "caring", "pride",
                         "admiration", "gratitude", "relief", "approval", "funny", "enthusiastic", "calm"],
            "negative": ["fearful", "frustrated", "nervousness", "remorse",
                         "embarrassed", "disappointed", "sad", "grief",
                         "disgusted", "angry", "annoyed", "disapproval", "hate", "worried", "bored", "empty"],
            "ambiguous": ["realization", "surprised", "curiosity", "confusion"],
            "neutral": ["neutral"],
            "unknown": ["unknown", "other"]
        }

        self._inverted_sentiment_dictionary = self._invert_dictionary(self._sentiment_dictionary)

    def recognise_emotion(self, wav_file_path=None, text_string=None):
        if wav_file_path is not None:
            wav_file = AudioSegment.from_file(file=wav_file_path, format="wav")
            if (wav_file.frame_rate != 16000) or (wav_file.channels != 1):
                print("ERROR: Input audio file has to have sampling rate 16000 and 1 channel")
                print("Your file has: \n\tSampling rate: "+str(wav_file.frame_rate)+"\n\tChannel number: "+str(wav_file.channels))
                return "ERROR, wrong audio file sampling rate or number of channels!"

        if wav_file_path is None and text_string is not None:
            print("Only text processing..")
            text_result = self._text_model(text_string, top_k=1)[0]["label"]
            text_result = self._text_model_dictionary[text_result]
            return text_result

        elif wav_file_path is not None and text_string is None:
            print("Only speech processing..")
            speech_result = self._speech_model.generate(wav_file_path, granularity="utterance", extract_embedding=False)
            speech_result = self._speech_model_dictionary[np.argmax(speech_result[0]['scores'])]
            return speech_result

        elif wav_file_path is None and text_string is None:
            print("ERROR: You have to provide either a wav_file_path or a text_string or both!")

        else:
            print("Text and speech processing..")
            speech_result = self._speech_model.generate(wav_file_path, granularity="utterance", extract_embedding=False)
            speech_result = self._speech_model_dictionary[np.argmax(speech_result[0]['scores'])]
            text_result = self._text_model(text_string, top_k=1)[0]["label"]
            text_result = self._text_model_dictionary[text_result]
            return self._decision_tree(text_result, speech_result)

    def _sent(self, emotion):
        return self._inverted_sentiment_dictionary[emotion]

    def _decision_tree(self, text, speech):
        if text == "other":
            text = "unknown"
        if speech == "other":
            speech = "unknown"

        if text == speech:
            return text
        elif text == "unknown":
            return speech
        elif speech == "unknown":
            return text
        elif text == "neutral":
            return speech
        elif speech == "neutral":
            return text + " " + speech
        elif self._sent(text) == self._sent(speech):
            return text + " " + speech
        elif (self._sent(text) == "negative" and self._sent(speech) == "positive") or (
                self._sent(text) == "positive" and self._sent(speech) == "negative"):
            return "unknown"
        elif self._sent(text) == "ambiguous" or self._sent(speech) == "ambiguous":
            return text + " " + speech
        else:
            print("ERROR!!!")

    @staticmethod
    def _invert_dictionary(dictionary: dict) -> dict:
        inverted_dictionary = {}
        for key in dictionary.keys():
            for value in dictionary[key]:
                inverted_dictionary[value] = key
        return inverted_dictionary


if __name__ == '__main__':
    """
    USAGE EXAMPLE
    """

    pc = ProcessChain()

    from read_dataset import ReadDataset
    #paths, labels, transcriptions = ReadDataset("../datasets").load_IEMOCAP(transcriptions=True)
    paths, labels = ReadDataset("../datasets").load_RAVDESS()

    k = 333
    for i in range(1):
        k+=111
        test_sample = k
        input_wav_path = paths[test_sample]
        #input_text = transcriptions[test_sample]

        #print("Input:\n\tPath: \t"+input_wav_path+"\n\tTranscription: \t"+input_text)
        #print("\nOutput:\t"+pc.recognise_emotion(wav_file_path=input_wav_path, text_string=input_text))

        print("Input:\n\tPath: \t" + input_wav_path)
        print("\nOutput:\t"+pc.recognise_emotion(wav_file_path=input_wav_path))
        print("Ground-truth: " + labels[test_sample])


