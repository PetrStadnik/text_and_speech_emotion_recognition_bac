import numpy as np
import os
from read_dataset import ReadDataset
from funasr import AutoModel
from analyze import Analyze
from process_wav import Process

def invert_dictionary(dictionary: dict) -> dict:
    inverted_dictionary = {}
    for key in dictionary.keys():
        for value in dictionary[key]:
            inverted_dictionary[value] = key
    return inverted_dictionary


model = AutoModel(model="iic/emotion2vec_base_finetuned", model_revision="v2.0.4")

"""
9-class emotions:
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
"""

sent_dict = {
        "positive": ["happy", "amusement", "excited", "joy", "love", "desire", "optimism", "caring", "pride",
                     "admiration", "gratitude", "relief", "approval", "funny", "enthusiastic", "calm"],
        "negative": ["fearful", "frustrated", "nervousness", "remorse",
                     "embarrassed", "disappointed", "sad", "grief",
                     "disgusted", "angry", "annoyed", "disapproval", "hate", "worried", "bored", "empty"],
        "ambiguous": ["realization", "surprised", "curiosity", "confusion", "other", "unknown"],
        "neutral": ["neutral"]
    }

inv_dict = invert_dictionary(sent_dict)

edict = {
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
permissible_emotions = ["angry", "happy", "neutral", "sad", "unknown", "fearful", "surprised", "disgusted", "other"]


paths, labels = ReadDataset(dataset_path="../datasets").load_IEMOCAP()
#analyzer = Analyze(edict, model_name="emotion2vec", dataset_name="IEMOCAP")

results = np.array([])
lbl = np.array([])

for i in range(len(paths)):
    #if labels[i] not in permissible_emotions:
        #continue
    if labels[i] == "other" or labels[i] == "unknown":
        continue

    inp = str(paths[i])
    inp = Process(inp, normalize=False, add_noise=False, reduce_noise=False).get_as_wav()
    rec_result = model.generate(inp, output_dir="./outputs", granularity="utterance", extract_embedding=False)
    results = np.append(results, edict[np.argmax(rec_result[0]['scores'])])
    #results = np.append(results, inv_dict[edict[np.argmax(rec_result[0]['scores'])]])
    lbl = np.append(lbl, labels[i])
    #lbl = np.append(lbl, inv_dict[labels[i]])
    #analyzer.add_value(edict[np.argmax(rec_result[0]['scores'])], labels[i])

#analyzer.show_analysis()
#analyzer.nice_table()

import matplotlib.pyplot as plt
import audplot

fig, axs = plt.subplots(1, 1, figsize=[7, 7])

axs.set_title('Model: emotion2vec | Dataset: IEMOCAP \n (normalize=True, add_noise=False, reduce_noise=False)')
#axs.set_title('Model: emotion2vec | Dataset: IEMOCAP \n SENTIMENT')
audplot.confusion_matrix(
    lbl,
    results,
    percentage=True,
    ax=axs,
)
plt.tight_layout()
plt.show()
#fig.savefig('tests/emoOVA_TFF.png')

np.save('tests/all_emo_sent_emotion2vecIEM_FFF_res.npy', results)
np.save('tests/all_emo_sent_emotion2vecIEM_FFF_lbl.npy', lbl)
