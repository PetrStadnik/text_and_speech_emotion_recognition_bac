import os
from read_dataset import ReadDataset
from analyze import Analyze
from process_wav import Process
import audeer
import audonnx
import numpy as np
import audinterface
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from joblib import dump, load


def download_model():
    audeer.mkdir(cache_root)

    def cache_path(file):
        return os.path.join(cache_root, file)

    url = 'https://zenodo.org/record/6221127/files/w2v2-L-robust-12.6bc4a7fd-1.1.0.zip'
    dst_path = cache_path('model.zip')

    if not os.path.exists(dst_path):
        audeer.download_url(
            url,
            dst_path,
            verbose=True,
        )

    if not os.path.exists(model_root):
        audeer.extract_archive(
            dst_path,
            model_root,
            verbose=True,
        )

def train_classificator(hidden_st):
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    dataset_path = "../datasets"
    dataset_reader = ReadDataset(dataset_path=dataset_path)
    print("Reading NORM-CREMA-D dataset...")
    paths, labels = dataset_reader.load_NORM_CREMA_D()
    features = hidden_st.process_files(paths)
    clf.fit(features, labels)

    dump(clf, 'norm_support_vectors.joblib')
    print("Classificator saved!")

def predict(features):
    clf = load('support_vectors.joblib')
    predicted = clf.predict(features)
    return predicted


if __name__ == '__main__':

    model_root = 'model'
    cache_root = 'cache'

    #download_model()
    model = audonnx.load(model_root)
    print(model)

    hidden_states = audinterface.Feature(
        model.labels('hidden_states'),
        process_func=model,
        process_func_args={
            'outputs': 'hidden_states',
        },
        sampling_rate=16000,
        resample=True,
        num_workers=5,
        verbose=True,
    )

    #train_classificator(hidden_states)

    #file_features = hidden_states.process_file(paths[i])
    #predict(file_features)
    edict = {
        "angry": "angry",
        "happy": "happy",
        "sad": "sad",
        "fearful": "fearful",
        "disgusted": "disgusted",
        "neutral": "neutral",
    }
    permissible_emotions = ["angry", "happy", "neutral", "sad", "fearful", "disgusted"]
    dataset_path = "../datasets"
    dataset_reader = ReadDataset(dataset_path=dataset_path)
    print("Reading dataset...")
    paths, labels = dataset_reader.load_emoMovieDB()
    analyzer = Analyze(edict, model_name="my_wav2vec", dataset_name="emoMovieDB")

    results = np.array([])
    lbl = np.array([])

    for i in range(len(paths)):
        if labels[i] not in permissible_emotions:
            continue
        inp = str(paths[i])
        inp = Process(inp, normalize=False, add_noise=False, reduce_noise=False).get_as_wav()
        result = predict(hidden_states.process_file(inp))
        result = result[0]
        results = np.append(results, result)
        lbl = np.append(lbl, labels[i])
        analyzer.add_value(result, labels[i])
        print(i)

    analyzer.show_analysis()
    analyzer.nice_table()

    import matplotlib.pyplot as plt
    import audplot

    fig, axs = plt.subplots(1, 1, figsize=[7, 7])

    axs.set_title('Model: my_wav2vec | Dataset: emoMovieDB \n (normalize=False, add_noise=False, reduce_noise=False)')
    audplot.confusion_matrix(
        lbl,
        results,
        percentage=True,
        ax=axs,
    )
    plt.tight_layout()
    plt.show()
    fig.savefig('tests/my_wav2vecMOV_FFF.png')
