
import numpy as np
from read_dataset import ReadDataset
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import torch
from analyze import Analyze
from process_wav import Process

dataset_path = "../datasets"
dataset_reader = ReadDataset(dataset_path=dataset_path)
print("Reading dataset...")
paths, labels = dataset_reader.load_emoDBova()
results = np.array([])
lbl = np.array([])

edict = {
    0: "angry",
    1: "calm",
    2: "disgusted",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised"
}
permissible_emotions = ["angry", "happy", "neutral", "sad", "surprised", "calm", "disgusted", "fearful"]
analyzer = Analyze(edict, model_name="wav2vec2", dataset_name="MOV")

model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

model.projector = torch.nn.Linear(1024, 1024, bias=True)
model.classifier = torch.nn.Linear(1024, 8, bias=True)

torch_state_dict = torch.load('wav2vec_model/snapshots/17cf17c4ec7c4d083a3ac50a3c98d44088434cee/pytorch_model.bin', map_location=torch.device('cpu'))

model.projector.weight.data = torch_state_dict['classifier.dense.weight']
model.projector.bias.data = torch_state_dict['classifier.dense.bias']

model.classifier.weight.data = torch_state_dict['classifier.output.weight']
model.classifier.bias.data = torch_state_dict['classifier.output.bias']

for i in range(len(paths)):
    if labels[i] not in permissible_emotions:
        continue
    audio_file = paths[i]
    sound = Process(audio_file, normalize=True, add_noise=False, reduce_noise=False).get_as_pydub()
    sound_array = np.array(sound.get_array_of_samples())

    input = feature_extractor(
        raw_speech=sound_array,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt")

    result = model.forward(input.input_values.float())
    result = result.logits.detach().numpy()[0]
    #print(result)
    results = np.append(results, edict[np.argmax(result)])
    lbl = np.append(lbl, labels[i])
    analyzer.add_value(edict[np.argmax(result)], labels[i])
    print(i)

analyzer.show_analysis()
analyzer.nice_table()

import matplotlib.pyplot as plt
import audplot

fig, axs = plt.subplots(1, 1, figsize=[7, 7])

axs.set_title('Model: ehcalabres_wav2vec2 | Dataset: emoDBova \n (normalize=True, add_noise=False, reduce_noise=False)')
audplot.confusion_matrix(
    lbl,
    results,
    percentage=True,
    ax=axs,
)
plt.tight_layout()
fig.savefig('tests/ehcalabresOVA_TFF.png')
plt.show()
