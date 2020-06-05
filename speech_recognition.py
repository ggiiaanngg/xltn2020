import pickle

import librosa
import numpy as np
import os
import math
from sklearn.cluster import KMeans
import hmmlearn.hmm
import sounddevice as sd
from scipy.io.wavfile import write
import os
import nltk
from tkinter import *
import time
import matplotlib.pyplot as plt

arr18 = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])

arr12 = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])

arr9 = np.array([
    [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])

arr6 = np.array([
    [0.6, 0.4, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.1, 0.9, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.2, 0.8, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
])

st18 = np.array([0.3, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0])
st12 = np.array([0.0, 0.6, 0.1, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1])
st9 = np.array([0.0, 0.5, 0.1, 0.0, 0.3, 0.1, 0.0, 0.0, 0.0])
st6 = np.array([0.0, 0.0, 0.1, 0.3, 0.6, 0.0])


def get_mfcc(file_path):
    y, sr = librosa.load(file_path)  # read .wav file
    hop_length = math.floor(sr * 0.010)  # 10ms hop
    win_length = math.floor(sr * 0.025)  # 25ms frame
    mfcc = librosa.feature.mfcc(
        y, sr, n_mfcc=12, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    mfcc = mfcc - np.mean(mfcc, axis=1).reshape((-1, 1))
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    X = np.concatenate([mfcc, delta1, delta2], axis=0)  # O^r
    return X.T


def get_class_data(data_dir):
    files = os.listdir(data_dir)
    mfcc = [get_mfcc(os.path.join(data_dir, f)) for f in files if f.endswith(".wav")]
    return mfcc


def clustering(X, n_clusters=7):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0, verbose=0)
    kmeans.fit(X)
    print("centers", kmeans.cluster_centers_.shape)
    return kmeans


def predict():
    if __name__ == "__main__":
        class_names = ["nguoi", "hon", "ra", "vietnam", "toi", "test"]
        dataset = {}
        models = {}

        # for cname in class_names:
        #     print(f"Load {cname} dataset")
        #     dataset[cname] = get_class_data(os.path.join("data", cname))

        dataset["test"] = get_class_data(os.path.join("data", "test"))
        all_vectors = np.concatenate([np.concatenate(v, axis=0) for k, v in dataset.items()], axis=0)
        print("vectors", all_vectors.shape)
        kmeans = clustering(all_vectors)
        print("centers", kmeans.cluster_centers_.shape)
        class_vectors = dataset["test"]
        dataset["test"] = list([kmeans.predict(v).reshape(-1, 1) for v in dataset["test"]])
        for cname in class_names:
            # class_vectors = dataset[cname]
            # dataset[cname] = list([kmeans.predict(v).reshape(-1, 1) for v in dataset[cname]])

            # if cname.__contains__("ra"):
            #     hmm = hmmlearn.hmm.MultinomialHMM(
            #         n_components=6, random_state=0, n_iter=1000, verbose=True,
            #         startprob_prior=np.array([0.7, 0.2, 0.1, 0.0, 0.0, 0.0]),
            #         transmat_prior=np.array([
            #             [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, ],
            #             [0.1, 0.1, 0.5, 0.1, 0.1, 0.1, ],
            #             [0.1, 0.1, 0.1, 0.5, 0.1, 0.1, ],
            #             [0.1, 0.1, 0.1, 0.1, 0.5, 0.1, ],
            #             [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, ],
            #             [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, ],
            #         ]),
            #     )
            # if cname.__contains__("nguoi"):
            #     hmm = hmmlearn.hmm.MultinomialHMM(
            #         n_components=9, random_state=0, n_iter=1000, verbose=True,
            #         startprob_prior=st9,
            #         transmat_prior= arr9,
            #     )
            # if cname.__contains__("hon"):
            #     hmm = hmmlearn.hmm.MultinomialHMM(
            #         n_components=9, random_state=0, n_iter=1000, verbose=True,
            #         startprob_prior=st9,
            #         transmat_prior=arr9,
            #     )
            # if cname.__contains__("toi"):
            #     hmm = hmmlearn.hmm.MultinomialHMM(
            #         n_components=9, random_state=0, n_iter=1000, verbose=True,
            #         startprob_prior=st9,
            #         transmat_prior=arr9,
            #     )
            # if cname.__contains__("vietnam"):
            #     hmm = hmmlearn.hmm.MultinomialHMM(
            #         n_components=18, random_state=0, n_iter=1000, verbose=True,
            #         startprob_prior= st18,
            #         transmat_prior= arr18,
            #     )
            # if cname != "test":
            #     # with open(os.path.join("dataset", cname + ".pkl"), "wb") as file:
            #     #     pickle.dump(dataset[cname], file)
            #     with open(os.path.join("dataset", cname + ".pkl"), "rb") as file:
            #         dataset[cname] = pickle.load(file)
            if cname[:4] != 'test':
                # X = np.concatenate(dataset[cname])
                # lengths = list([len(x) for x in dataset[cname]])
                # print("training class", cname)
                # print(X.shape, lengths, len(lengths))
                # hmm.fit(X, lengths=lengths)
                # models[cname] = hmm
                # with open(os.path.join("models", cname + ".pkl"), "wb") as file:
                #     pickle.dump(models[cname], file)
                with open(os.path.join("models", cname + ".pkl"), "rb") as file:
                    models[cname] = pickle.load(file)
    print("Training done")

    print("Testing")

    correct = 0
    all = 0
    print("------------------------------------------------------")
    for O in dataset["test"]:
        all = all + 1
        score = {cname: model.score(O, [len(O)]) for cname, model in models.items() if cname[:4] != 'test'}
        print("test", score)
        max_value = max(score.values())
        max_key = [k for k, v in score.items() if v == max_value]
        max_key = str(max_key).split("'")[1]
        text.delete(1.0, END)
        text.insert(INSERT, max_key)
        text.pack()
        print(max_key)
    #     if "test".__contains__(max_key):
    #         correct = correct + 1
    # accuracy = correct / all
    # print("Accuracy = " + str(accuracy))


path = r"data/test"
fs = 44100
seconds = 60
gui = Tk(className='Predict word')
gui.geometry("200x200")
text = Text(gui)
index = 0
startTime = 0


def start_rec():
    global index
    global startTime
    startTime = time.time()
    time.sleep(0.3)
    text.delete(1.0, END)
    text.pack()
    index = index + 1
    global recording
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    return startTime


def stop_rec():
    text.delete(1.0, END)
    text.insert(INSERT, "Predicting...")
    text.pack()
    sd.stop()
    duration = time.time() - startTime - 0.3
    frame = int(duration * fs)
    for root, dirs, files in os.walk("data/test"):
        for file in files:
            os.remove(os.path.join(root, file))
    write(path + '/' + str(len(os.listdir(path))) + '.' + 'wav', fs, recording[:frame])
    predict()


btnRecord = Button(gui, text='Record', command=start_rec)
btnRecord.pack()
btnSave = Button(gui, text='Predict', command=stop_rec)
btnSave.pack()
# predict()
gui.mainloop()
