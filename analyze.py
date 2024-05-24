import numpy as np
import texttable
import seaborn as sns
import pandas as pd
from datetime import datetime


class Analyze:

    def __init__(self, emotion_dict: dict, model_name: str, dataset_name: str) -> None:
        self.edict = emotion_dict
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.s = np.unique(np.array(list(self.edict.values()))).tolist()
        #print(self.s)
        lst = []
        for i in range(len(self.s)):
            lst.append((self.s[i], i))
        self.dct = dict(lst)
        self.table = self.table = np.zeros((len(self.s), len(self.s)))

        self.total_correct = 0
        self.total_incorrect = 0
        self.samp = np.zeros((len(self.s)))


    def get_table(self):
        return self.table

    def add_value(self, prediction, correct):
        self.table[self.dct[correct]][self.dct[prediction]] +=1
        self.samp[self.dct[correct]] += 1
        if correct == prediction:
            self.total_correct += 1
        else:
            self.total_incorrect += 1

    def show_analysis(self):
        matching_table = texttable.Texttable()

        matching_table.add_row(["emotions | prediction-->"]+self.s)

        for i in range(len(self.s)):
            matching_table.add_row([self.s[i]] + self.table[i][:].tolist())

        print(matching_table.draw())

        total = self.total_correct + self.total_incorrect
        print("Total samples: " + str(total))
        dataset_table = texttable.Texttable()
        dataset_table.add_row(list(self.s))
        dataset_table.add_row(self.samp.tolist())
        print(dataset_table.draw())
        print("Total correct: " + str(self.total_correct) + "\t Total incorrect: " + str(self.total_incorrect))
        print("Success: " + str("{:.2f}".format(100*self.total_correct/total)))

    def nice_table(self):

        df = pd.DataFrame(self.table,
                          index=pd.Index(self.s, name='Actual Label:'),
                          columns=pd.MultiIndex.from_product([[self.model_name + " (" + self.dataset_name+")"],self.s], names=['Model (Dataset):', 'Predicted:']))



        # Creating an HTML file
        dt = str(datetime.now())
        dt = dt.replace(':', '_').replace(' ','_').replace('.','')
        Func = open("html_results/"+self.model_name+"_"+self.dataset_name+"_"+dt+".html", "w")

        # Adding input data to the HTML file
        Func.write("<html>\n" + df.style.background_gradient(axis=1).format(precision=0).to_html() + "\n</html>")
        # Saving the data into the HTML file
        Func.close()



if __name__ == '__main__':

    """
    EXAMPLE OF USE
    """
    edict = {
        0: "angry",
        3: "happy",
        6: "sad",
        8: "other",
        2: "fearful",
        7: "surprised",
        1: "disgusted",
        4: "neutral",
        5: "other",
    }

    an = Analyze(edict, "test_model", "test_dataset")

    an.add_value("happy", "happy")
    an.add_value("fearful", "fearful")
    an.add_value("surprised", "sad")
    an.add_value("surprised", "sad")
    an.add_value("disgusted", "sad")
    an.add_value("disgusted", "sad")
    an.add_value("disgusted", "sad")
    an.add_value("disgusted", "sad")

    an.show_analysis()
    an.nice_table()

