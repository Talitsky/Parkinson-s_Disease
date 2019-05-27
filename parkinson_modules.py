import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal as signal
from scipy.stats import skew, kurtosis
import os

class parkinson():
    def __init__(self, path):
        self.path = path
        self.exercises = {}
        self.time = {}
        
    def read(self, index = list(range(20))):
        '''
        read all exercise
        patient info - part of info excel with patient 
        index minus/plus        
        '''
        os.chdir(self.path)
        file_list = os.listdir()
        for file in file_list:
            if (file[-4:] == '.csv'):
                try:
                    ex = pd.read_csv(file) 
                    if (file[3] != '_'):
                        i = 10 + int(file[3])
                    else:
                        i = int(file[2])
                    t = ex["time, s"]
                    j = index[i-1]
                    if j > 0:
                        self.time[j] = t
                        self.exercises[j] = {}
                        self.exercises[j]['acc_x, mg'] = ex['acc_x, mg']
                        self.exercises[j]['acc_y, mg'] = ex['acc_y, mg']
                        self.exercises[j]['acc_z, mg'] = ex['acc_z, mg']
                        self.exercises[j]['gyr_x, dps'] = ex['gyr_x, dps']
                        self.exercises[j]['gyr_y, dps'] = ex['gyr_y, dps']
                        self.exercises[j]['gyr_z, dps'] = ex['gyr_z, dps']
                        self.exercises[j]['mag_x, mga'] = ex['mag_x, mga']
                        self.exercises[j]['mag_y, mga'] = ex['mag_y, mga']
                        self.exercises[j]['mag_z, mga'] = ex['mag_z, mga']
                except:
                    print(file, '\n', '!!ERROR!!')
                    pass
            
    def to_dict(self, list_exercises = True, timing = True):
        '''
        Add exercise to pandas file.
        Input:
        file - pandas file
        list_exercise - list of all exercise / can be True then all of exercises will be added
        timing - list of start and end of each exercise/ can be True, then time series won't be cut
        '''
        features = {}
        if list_exercises == True:
            list_exercises = np.arange(len(self.exercises))
        for i in self.exercises:
            for key in self.exercises[i]:
                if timing == True:
                    features.update(calculate_feature(self.time[i], 
                                                 self.exercises[i][key], label = str(i) +' '+ key))
                else:
                    index_initial = np.argwhere(self.time[i] > timing[i][0])
                    index_end = np.argwhere(self.time[i] > timing[i][1])
                    features.update(calculate_feature(self.time[i][index_initial:index_end],
                                                 self.exercises[i][key][index_initial:index_end], label = str(i) +' '+ key))
#         file.append(features)
        return features
                
                
def calculate_feature(time, data, 
                      label = True, calculate_fourier_feature = True):
    
    feature = {}
    if label == True:
        label = ''
    else:
        label += '_'
    feature[label + 'std'] = np.std(data.values)
    feature[label + 'mean'] = np.mean(data.values)
    feature[label + 'skew'] = skew(data.values)
    feature[label + 'skew'] = kurtosis(data.values)
    data = (data - np.mean(data.values))/(np.std(data.values) + 1.e-3)
    b = time.values[1:] - time.values[0:-1]
    feature[label + 'differential_mean'] = np.mean(np.divide(data.values[1:] - data.values[:-1], b, where=b!=0))
    feature[label + 'differential_std' ] = np.std( np.divide(data.values[1:] - data.values[:-1], b, where=b!=0))
    N = 10
    data_filt = data.rolling(window = N + 1).mean()
#     print(data_filt.shape)
    std = (data - data_filt)[N:]
    
#     print(std.values.shape)
    feature[label + 'std_without_trend'] = np.std(std)
    feature[label + 'mean_without_trend'] = np.mean(std)
    if calculate_fourier_feature:
        feature.update(calculate_fft_feature(time.values.reshape(-1)[N:],
                                             std.values.reshape(-1)[N:], label = label + 'noise_'))
        feature.update(calculate_fft_feature(time.values.reshape(-1)[N:],
                                             data_filt.values.reshape(-1)[N:], label = label + 'trend_'))
    return feature
            
def calculate_fft_feature(time, data, label, threshold = 3):
    feature_fft = {}
    timestamp = (time[len(time)-1] - time[0])/len(time)
    fft_data = np.abs(np.fft.fft(data))
    fft_freq = np.fft.fftfreq(len(fft_data), timestamp)
    peaks, _ = signal.find_peaks(fft_data, fft_data.mean() + 2*np.std(fft_data))
    labels = np.argsort(fft_freq)
    peaks_label = np.argsort(np.abs(fft_freq[peaks]))
    freqs = np.abs(fft_freq[peaks])[peaks_label] [np.abs(fft_freq[peaks])[peaks_label] > threshold]
    ampls = np.abs(fft_data[peaks])[peaks_label] [np.abs(fft_freq[peaks])[peaks_label] > threshold]
    if len(freqs) != 0:
        feature_fft[label + 'peaks_freq_mean'] = np.mean(freqs)
        feature_fft[label + 'peaks_freq_std'] = np.std(freqs)
    else:
        feature_fft[label + 'peaks_freq_mean'] = 0
        feature_fft[label + 'peaks_freq_std'] = 0
        
    if len(ampls) !=0:
        feature_fft[label + 'peaks_amplitude_mean'] = np.mean(ampls)
        feature_fft[label + 'peaks_amplitude_std'] = np.std(ampls)
    else:
        feature_fft[label + 'peaks_amplitude_mean'] = 0
        feature_fft[label + 'peaks_amplitude_std'] = 0
    feature_fft[label + 'spectrum_energy_mean'] = np.mean(fft_data**2)
    feature_fft[label + 'spectrum_energy_std'] = np.std(fft_data**2)
    return feature_fft
                
def plot_mean_and_variance(X, time, l = 1000, N = 10):
    x_filt = X.rolling(window=N + 1)
    x_filt_mean = x_filt.mean()
    std_h = (X - x_filt_mean)[N:]
    std = np.std(std_h)
    plt.figure(0, figsize=(10,5))
    plt.plot(time[:l], x_filt_mean[N:N+l], c = 'r', label = 'Smooth Signal')
    plt.fill_between(time[:l], x_filt_mean[N:N+l] - 2*std, x_filt_mean[N:N+l] + 2*std , color = 'grey', alpha = 0.2, 
                     label = '95% CI, $\sigma$ =' + str(round(std)))
    plt.plot(time[:l], X[:l], c = 'b', label = 'Real Signal', alpha = 0.4)
