import numpy as np
import os
PATH = "../datasets/IEMOCAP_full_release_withoutVideos/IEMOCAP_full_release/"

edict = {
    "ang": "angry",
    "hap": "happy",
    "exc": "excited",
    "sad": "sad",
    "fru": "frustrated",
    "fea": "fearful",
    "sur": "surprised",
    "dis": "disgusted",
    "neu": "neutral",
    "oth": "other",
}

paths = np.array([])
labels = np.array([])

filenames = np.array([])

class Read_iemocap:
    def __init__(self, dataset_path):
        self.dspath = dataset_path

        self.pathes_to_file = np.array([])
        self.filenames = np.array([])
        self.transcriptions_dict = {}
        self.labels = np.array([])

    def load_dataset(self):
        print("Loading IEMOCAP dataset...")
        for i in range(1, 6):
            self.prepare_session(i)
            #print(i)

        transcriptions_arr = np.array([])
        for name in self.filenames:
            transcriptions_arr = np.append(transcriptions_arr, self.transcriptions_dict[name])

        print("Dataset successfully loaded with " + str(len(self.labels)) + " items.")

        return self.pathes_to_file, self.labels, transcriptions_arr


    def prepare_session(self, ses):
        path = self.dspath + 'Session'+str(ses)+'/dialog/EmoEvaluation/'
        for filename in os.listdir(path):
             if os.path.isfile(path + filename) and filename.split('.')[-1] == 'txt':
                #print(path+filename)
                with open(path+filename) as f:
                    for line in f.readlines():
                        if line.startswith('['):
                            a = line.split("\t")
                            if a[2] == "xxx":
                                continue
                            else:
                                self.filenames = np.append(self.filenames, a[1])
                                self.labels = np.append(self.labels, edict[a[2]])
                                self.pathes_to_file = np.append(self.pathes_to_file, PATH+"Session"+str(ses)+"/sentences/wav/"+filename.split('.')[0] + "/" + a[1]+".wav")

                with (open(self.dspath + 'Session'+str(ses)+'/dialog/transcriptions/'+filename) as f):
                    for line in f.readlines():
                        #print(line)
                        a = line.split(" ", 2)
                        #self.pathes_to_file = np.append(self.pathes_to_file, "")
                        # self.filenames = np.append(self.filenames, a[0])
                        if a[0] in self.filenames:
                            self.transcriptions_dict[a[0]] = a[2].split("\n")[0].replace("[GARBAGE]", ""
                                                                                         ).replace("[BREATHING]", ""
                                                                                                   ).replace("[LAUGHTER]", ""
                                                                                                             ).replace("[garbage]", ""
                                                                                                                       ).replace("[laughter]", "")





if __name__ == '__main__':
    r = Read_iemocap(PATH)
    paths, labels, transcriptions = r.load_dataset()
    for i in range(len(r.labels)):
        print(paths[i] + " = " + labels[i] + " = " + transcriptions[i])

    print(len(paths))
