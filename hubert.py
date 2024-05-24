from transformers import pipeline
from read_dataset import ReadDataset
from process_wav import Process
from analyze import Analyze
import numpy as np
if __name__ == '__main__':

    edict = {
        "ang": "angry",
        "hap": "happy",
        "neu": "neutral",
        "sad": "sad",
    }
    permissible_emotions = ["angry", "happy", "neutral", "sad"]
    paths, labels = ReadDataset("../datasets").load_emoDBova(veracity=80)

    analyzer = Analyze(edict, model_name="s3prl_hubert", dataset_name="OVA")

    results = np.array([])
    lbl = np.array([])

    classifier = pipeline("audio-classification", model="superb/hubert-base-superb-er")

    for i in range(len(paths)):
        if labels[i] not in permissible_emotions:
            continue
        else:
            result = edict[classifier(Process(paths[i], normalize=False).get_as_wav(p="edited/hubertedit.wav"), top_k=1)[0]["label"]]
            results = np.append(results, result)
            lbl = np.append(lbl, labels[i])
            analyzer.add_value(result, labels[i])

    analyzer.show_analysis()
    analyzer.nice_table()

    import matplotlib.pyplot as plt
    import audplot

    fig, axs = plt.subplots(1, 1, figsize=[7, 7])

    axs.set_title(
        'Model: s3prl_hubert | Dataset: emoDBova \n (normalize=False, add_noise=False, reduce_noise=False)')
    audplot.confusion_matrix(
        lbl,
        results,
        percentage=True,
        ax=axs,
    )
    plt.tight_layout()
    plt.show()
    fig.savefig('tests/s3prl_hubertOVA_FFF.png')


