import numpy as np
from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise
from pydub.playback import play
import matplotlib.pyplot as plt
import noisereduce

#from read_dataset import ReadDataset


class Process:
    def __init__(self, wav_url, sampling_rate=16000, normalize=False, add_noise=False, reduce_noise=False):
        self.wav_url = wav_url
        self.edited = ""
        self.wav_file = AudioSegment.from_file(file=wav_url, format="wav")

        if self.wav_file.channels > 1:
            self.wav_file = self.wav_file.set_channels(1)

        if add_noise:
            gain = self.get_loudness() - 10.0
            self.wav_file = self.wav_file.overlay(
                WhiteNoise(sample_rate=self.wav_file.frame_rate).to_audio_segment(duration=len(self.wav_file)).apply_gain(gain))

        if sampling_rate is not None:
            self.wav_file = self.wav_file.set_frame_rate(sampling_rate)

        if reduce_noise:
            reduced = noisereduce.reduce_noise(y=self.wav_file.get_array_of_samples(), sr=self.wav_file.frame_rate)
            self.wav_file = AudioSegment(np.int16(reduced).tobytes(), frame_rate=self.wav_file.frame_rate,
                                         sample_width=self.wav_file.sample_width, channels=1)

        if normalize:
            self.wav_file = effects.normalize(self.wav_file)

    def get_loudness(self):
        return self.wav_file.dBFS

    def get_max_amplitude(self):
        return self.wav_file.max

    def get_duration(self):
        return len(self.wav_file) / 1000

    def play(self, start=0, end=None):
        if end is None:
            play(self.wav_file)
        else:
            play(self.wav_file[start:end])
    def get_as_wav(self, p="edited/edited.wav"):
        self.wav_file.export(out_f=p, format="wav")
        return p

    def show_info(self):
        print("wav_url: {} ".format(self.wav_url))
        print("Frame rate [Hz]: " + str(self.wav_file.frame_rate))
        print("Channels: " + str(self.wav_file.channels))
        print("Max amplitude: " + str(self.wav_file.max))
        print("Length [s] : " + str(len(self.wav_file) / 1000))
        print("Loudness: " + str(self.get_loudness()))

    def get_part(self, begin, end):
        return self.wav_file[begin:end]

    def get_part_as_wav(self, begin, end, p="edited/edited.wav"):
        return self.wav_file[begin:end].export(out_f=p, format="wav")

    def get_as_pydub(self):
        return self.wav_file

    def get_as_nparray(self):
        return np.array(self.wav_file.get_array_of_samples())

    def plot(self):

        time = np.arange(0, len(self.wav_file.get_array_of_samples()), 1) / self.wav_file.frame_rate

        plt.title(self.wav_url)
        plt.ylabel('Amplitude')
        plt.xlabel('Time(seconds)')
        plt.plot(time, self.wav_file.get_array_of_samples(), linewidth=0.2, alpha=0.7, )
        plt.show()

    def plot_with_spectogram(self):
        time = np.arange(0, len(self.wav_file.get_array_of_samples()), 1) / self.wav_file.frame_rate
        signal_data = self.wav_file.get_array_of_samples()
        sampling_frequency = self.wav_file.frame_rate
        plt.figure(1, figsize=(8, 6))
        plt.subplot(211)
        #plt.title('File {}'.format(self.wav_url))
        #plt.title("After white noise added")
        plt.title("After noise reduced")
        plt.plot(time, self.wav_file.get_array_of_samples(), linewidth=0.3, alpha=1, )
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.subplot(212)
        plt.specgram(signal_data, Fs=sampling_frequency)
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()


"""
t1 = t1 * 1000 #Works in milliseconds
t2 = t2 * 1000
newAudio = AudioSegment.from_wav("oldSong.wav")
newAudio = newAudio[t1:t2]
newAudio.export('newSong.wav', format="wav") #Exports to a wav file in the current path.
"""

if __name__ == '__main__':

    process = Process('../datasets/CREMA-D/AudioWAV/1001_IEO_HAP_HI.wav', normalize=True, add_noise=False, reduce_noise=True)
    #process = Process('../datasets/TESS/YAF_germ_happy.wav', normalize=True)
    process.show_info()
    #process.plot()
    process.plot_with_spectogram()
    #process.play()
    #process.get_as_wav()
    print("=================================================")
    #process = Process('../datasets/RAVDESS/Actor_04/03-01-01-01-02-02-04.wav', normalize=True, reduce_noise=True)
    #process.show_info()
    #process.plot()
    #process.play()
    #process.get_as_wav()


