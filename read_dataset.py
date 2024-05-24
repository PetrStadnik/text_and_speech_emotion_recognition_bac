import numpy as np
import csv
import os
import re
import matplotlib.pyplot as plt
from read_iemocap import Read_iemocap
from process_wav import Process

class ReadDataset:
    def __init__(self, dataset_path):
        self.paths = np.array([])
        self.labels = np.array([])
        self.dataset_path = dataset_path

    def load_IEMOCAP(self, transcriptions=False):
        PATH = "../datasets/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/"
        r = Read_iemocap(PATH)
        paths, labels, transcript = r.load_dataset()
        if transcriptions:
            return paths, labels, transcript
        else:
            return paths, labels


    def load_RAVDESS(self):
        edict = {
            "05": "angry",
            "03": "happy",
            "04": "sad",
            "06": "fearful",
            "08": "surprised",
            "07": "disgusted",
            "01": "neutral",
            "02": "calm",
        }
        for dirname, _, filenames in os.walk(self.dataset_path+'/RAVDESS'):
            for filename in filenames:
                label = filename.split('-')[2]
                lb = edict[label.lower()]
                self.labels = np.append(self.labels, lb)
                self.paths = np.append(self.paths, (os.path.join(dirname, filename)).replace("\\", "/"))

        print('RAVDESS loaded, total elements: ', len(self.labels))
        return self.paths, self.labels


    def load_TESS(self):
        edict = {
            "angry": "angry",
            "happy": "happy",
            "sad": "sad",
            "fear": "fearful",
            "ps": "surprised",
            "disgust": "disgusted",
            "neutral": "neutral",
        }

        for dirname, _, filenames in os.walk(self.dataset_path+'/TESS'):
            for filename in filenames:
                label = filename.split('_')[-1]
                label = label.split('.')[0]
                lb = edict[label.lower()]
                #if lb == "calm" or lb == "surprised":
                    #continue
                self.labels = np.append(self.labels, edict[label.lower()])
                self.paths = np.append(self.paths, (os.path.join(dirname, filename)).replace("\\", "/"))
        print('TESS loaded, total elements: ', len(self.labels))
        return self.paths, self.labels

    def load_SAVEE(self):
        edict = {
            "a": "angry",
            "h": "happy",
            "sa": "sad",
            "f": "fearful",
            "su": "surprised",
            "d": "disgusted",
            "n": "neutral",
        }

        for dirname, _, filenames in os.walk(self.dataset_path+'/SAVEE'):
            for filename in filenames:
                label = filename.split('_')[-1]
                label = label.split('.')[0]
                temp = re.compile("([a-zA-Z]+)([0-9]+)")
                label = temp.match(label).groups()[0]
                lb = edict[label.lower()]
                #if lb == "calm" or lb == "surprised":
                    #continue
                self.labels = np.append(self.labels, edict[label.lower()])
                self.paths = np.append(self.paths, (os.path.join(dirname, filename)).replace("\\", "/"))
        print('SAVEE loaded, total elements: ', len(self.labels))
        return self.paths, self.labels

    def load_CREMA_D(self):
        edict = {
            "ANG": "angry",
            "HAP": "happy",
            "SAD": "sad",
            "FEA": "fearful",
            "DIS": "disgusted",
            "NEU": "neutral",
        }

        for dirname, _, filenames in os.walk(self.dataset_path+'/CREMA-D'):
            for filename in filenames:
                self.paths = np.append(self.paths, (os.path.join(dirname, filename)).replace("\\", "/"))
                label = filename.split('_')[2]
                self.labels = np.append(self.labels, edict[label])
        print('CREMA-D loaded, total elements: ', len(self.labels))
        return self.paths, self.labels

    def load_NORM_CREMA_D(self):
        edict = {
            "ANG": "angry",
            "HAP": "happy",
            "SAD": "sad",
            "FEA": "fearful",
            "DIS": "disgusted",
            "NEU": "neutral",
        }

        for dirname, _, filenames in os.walk(self.dataset_path+'/NORM-CREMA-D'):
            for filename in filenames:
                self.paths = np.append(self.paths, (os.path.join(dirname, filename)).replace("\\", "/"))
                label = filename.split('_')[2]
                self.labels = np.append(self.labels, edict[label])
        print('NORM-CREMA-D loaded, total elements: ', len(self.labels))
        return self.paths, self.labels

    def load_emoMovieDB(self):
        edict = {
            '1': "neutral",
            '2': "angry",
            '3': "sad",
            '4': "happy",
            '5': "fearful"
        }

        count = 0
        for dirname, _, filenames in os.walk(self.dataset_path + '/emoMovieDB/wav'):
            for filename in filenames:
                label = filename.split('_')[0]
                self.labels = np.append(self.labels, edict[label])
                self.paths = np.append(self.paths, (os.path.join(dirname, filename)).replace("\\", "/"))
                count += 1
        print('emoMovieDB loaded, total elements: ', count)

        return self.paths, self.labels

    def load_emoDBova(self, veracity=80.):
        with open(self.dataset_path + '/emoDBova/read_data.csv', "r", newline='', ) as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["id", "file_name", "veracity", "gender", "emotion"], delimiter=';')
            count = 0
            for row in reader:
                #print(row)
                if float(row['veracity'].replace(",",".")) >= float(veracity):
                    self.labels = np.append(self.labels, row["emotion"])
                    self.paths = np.append(self.paths, self.dataset_path + "/emoDBova/wav/" + row["file_name"] + ".wav")
                    count += 1
        print('emoDBova loaded, total elements: ', count)
        return self.paths, self.labels


if __name__ == '__main__':

    """
    USAGE EXAMPLE
    """
    dataset_path = "../datasets"
    dataset_reader = ReadDataset(dataset_path=dataset_path)
    print("Reading dataset...")
    paths, labels = dataset_reader.load_emoDBova(varacity=80)
    print(type(paths))
    print(type(paths[0]))
    Process(paths[0]).show_info()

    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))
    print(np.sum(counts))
    plt.figure(1, figsize=(18, 11))
    plt.subplot(221)
    plt.title("Emotion histogram")
    plt.ylabel("Number of records")
    plt.xlabel("Emotions")
    plt.bar(unique, counts)

    durations = np.array([])
    max_amp = np.array([])
    loudness = np.array([])

    for path in paths:
        pro = Process(path, normalize=False)
        durations = np.append(durations, pro.get_duration())
        max_amp = np.append(max_amp, pro.get_max_amplitude())
        loudness = np.append(loudness, pro.get_loudness())

    print(durations.mean())

    plt.subplot(222)
    plt.title("Duration histogram")
    plt.ylabel("Number of records")
    plt.xlabel("Durations in seconds")
    plt.hist(durations, bins=50, label="Duration histogram", range=(0, 20))

    plt.subplot(223)
    plt.title("Max amplitude histogram")
    plt.ylabel("Number of records")
    plt.xlabel("Maximal amplitude")
    plt.hist(max_amp, bins=50, label="Duration histogram", range=(0, 50000))

    plt.subplot(224)
    plt.title("Loudness histogram")
    plt.ylabel("Number of records")
    plt.xlabel("Loudness dB relative to full scale ")
    plt.hist(loudness, bins=50, label="Duration histogram", range=(-100, 5))

    plt.show()

