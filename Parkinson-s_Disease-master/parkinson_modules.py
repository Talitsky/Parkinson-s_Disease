import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.signal as signal
from scipy.stats import skew, kurtosis
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from scipy.signal import butter, welch, filtfilt
sclr = RobustScaler()
columns_acc = [
    "acc_x, mg",
    'acc_y, mg',
    'acc_z, mg'  
]
columns_hyr = [
    'gyr_x, dps',
    'gyr_y, dps',
    'gyr_z, dps'
]

columns_mag = [
    'mag_x, mga',
    'mag_y, mga',
    'mag_z, mga'
]


no_use_this_patients_per_exercise = {
 1: [22, 38, 57, 64, 25, 31, 32, 40, 55],
 2: [22, 38, 34, 53, 54, 55],
 3: [22, 38, 34, 53, 54, 55],
 4: [22, 38, 34, 53, 54, 55],
 5: [22, 38, 34, 53, 54, 55],
 6: [22, 38, 34, 53, 54, 55],
 7: [22, 38, 34, 53, 54, 55],
 8: [38, 57, 64, 25, 31, 32, 40, 55],
 9: [22, 38, 34, 53, 54, 55],
 10: [22, 38, 34, 53, 54, 55],
 11: [22, 38, 34, 53, 54, 55],
 12: [22, 38, 34, 53, 54, 57, 64, 25, 31, 32, 40, 55],
 13: [38, 35, 59, 54],
 14: [22, 38, 34, 53, 54],
 15: [22, 38, 34, 53, 54]    
}


class parkinson():
    '''
    parkinson Class
    to read data from patient folder
    '''
    def __init__(self, path):
        self.path = path
        self.exercises = {}
        self.time = {}
        
    def read(self, index = list(range(20)), filtering = True):
        '''
        read all exercise
        patient info - part of info excel with patient 
        index minus/plus        
        input indexes: indexes of exercises
        
        '''
        os.chdir(self.path)
        file_list = os.listdir()
        for file in file_list:
            if (file[-4:] == '.csv'):
                if '9a' in file:
                    continue
                try:
                    ex = pd.read_csv(file) 
                    if (file[3] != '_'):
                        i = 10 + int(file[3])
                    else:
                        i = int(file[2])
                    t = ex["time, s"]
                    j = i
                    if j > 0:
                        self.time[j] = t
                        self.exercises[j] = {}
                        if filtering:
                            self.exercises[j]['acc_x, mg']  = HPfiltering(ex['acc_x, mg'])
                            self.exercises[j]['acc_y, mg']  = HPfiltering(ex['acc_y, mg'])
                            self.exercises[j]['acc_z, mg']  = HPfiltering(ex['acc_z, mg'])
                            self.exercises[j]['gyr_x, dps'] = HPfiltering(ex['gyr_x, dps'])
                            self.exercises[j]['gyr_y, dps'] = HPfiltering(ex['gyr_y, dps'])
                            self.exercises[j]['gyr_z, dps'] = HPfiltering(ex['gyr_z, dps'])
                            self.exercises[j]['mag_x, mga'] = HPfiltering(ex['mag_x, mga'])
                            self.exercises[j]['mag_y, mga'] = HPfiltering(ex['mag_y, mga'])
                            self.exercises[j]['mag_z, mga'] = HPfiltering(ex['mag_z, mga'])
                        else:
                            self.exercises[j]['acc_x, mg']  = ex['acc_x, mg']
                            self.exercises[j]['acc_y, mg']  = ex['acc_y, mg']
                            self.exercises[j]['acc_z, mg']  = ex['acc_z, mg']
                            self.exercises[j]['gyr_x, dps'] = ex['gyr_x, dps']
                            self.exercises[j]['gyr_y, dps'] = ex['gyr_y, dps']
                            self.exercises[j]['gyr_z, dps'] = ex['gyr_z, dps']
                            self.exercises[j]['mag_x, mga'] = ex['mag_x, mga'] - np.mean(ex['mag_x, mga'])
                            self.exercises[j]['mag_y, mga'] = ex['mag_y, mga'] - np.mean(ex['mag_y, mga'])
                            self.exercises[j]['mag_z, mga'] = ex['mag_z, mga'] - np.mean(ex['mag_z, mga'])
                except:
                    print(file, '\n', '!!ERROR!!')
                    pass
            
    def to_dict(self, list_exercises = True, timing = True):
        '''
        Input:
        file - pandas file
        list_exercise - list of all exercise / can be True then all of exercises will be added
        timing - list of start and end of each exercise/ can be True, then time series won't be cut
        Output:
        features - dct of features
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
        return features
                
        
def HPfiltering(rawdata,cutoff=0.75,ftype='highpass'):
    '''
raw_data - pd.Series()
output   - pd.Series()
highpass (or lowpass) filter data. HP to remove gravity (offset - limb orientation) from accelerometer data from each visit (trial)
input: Activity dictionary, cutoff freq [Hz], task, sensor location and type of filter (highpass or lowpass).
'''
    idx = rawdata.index
    idx = idx-idx[0]
    rawdata.index = idx
    x = rawdata.values
    Fs = 100 #sampling rate
    #filter design
    cutoff_norm = cutoff/(0.5*Fs)
    b,a = butter(4,cutoff_norm,btype=ftype,analog=False)
    #filter data
    xfilt = filtfilt(b,a,x,axis=0)
    rawdatafilt = pd.Series(data=xfilt,index=rawdata.index)

    return rawdatafilt
                
def calculate_feature(time, data, 
                      label = True, calculate_fourier_feature = True):
    '''
    Calculate features:
    Input:
    time - array time
    data - pd.Series
    Output:
    Features - dict
    '''
    
    feature = {}
    if label == True:
        label = ''
    else:
        label += '_'
    feature[label + 'std'] = np.std(data.values)
    feature[label + 'mean'] = np.mean(data.values)
    feature[label + 'max'] = np.max(data.values)
    feature[label + 'min'] = np.min(data.values)
    
    feature[label + 'skew'] = skew(data.values)
    feature[label + 'kurtosis'] = kurtosis(data.values)
    data = (data - np.mean(data.values))/(np.std(data.values) + 1.e-3)
    b = time.values[1:] - time.values[0:-1]
    feature[label + 'differential_mean'] = np.mean(np.divide(data.values[1:] - data.values[:-1], b, where = b!=0))
    feature[label + 'differential_std' ] = np.std( np.divide(data.values[1:] - data.values[:-1], b, where = b!=0))
    N = 10
    data_filt = data.rolling(window = N + 1).mean()
    std = (data - data_filt)[N:]
    
    feature[label + 'noise_std' ] = np.std(std)
    feature[label + 'noise_mean'] = np.mean(std)
    if calculate_fourier_feature:
        feature.update(calculate_fft_feature(time.values.reshape(-1)[N:],
                                             std.values.reshape(-1)[N:], 
                                             label = label + 'noise_'))
        
        feature.update(calculate_fft_feature(time.values.reshape(-1)[N:],
                                             data_filt.values.reshape(-1)[N:], 
                                             label = label + 'trend_'))
    return feature

def add_fourier_features(freqs, ampls, feature_fft, label = ''):
    if len(freqs) != 0:
        feature_fft[label + 'peaks_freq_mean'] = np.mean(freqs)
        feature_fft[label + 'peaks_freq_std'] = np.std(freqs)
        
        feature_fft[label + 'peaks_freq_min'] = np.min(freqs)
        feature_fft[label + 'peaks_freq_max'] = np.max(freqs)

        feature_fft[label + 'dominant_frequency'] = freqs[0]
    else:
        feature_fft[label + 'peaks_freq_mean'] = 0
        feature_fft[label + 'peaks_freq_std'] = 0
        
        feature_fft[label + 'peaks_freq_min'] = 0
        feature_fft[label + 'peaks_freq_max'] = 0
        
        feature_fft[label + 'dominant_frequency'] = 0
        
    if len(ampls) !=0:
        feature_fft[label + 'peaks_amplitude_mean'] = np.mean(ampls)
        feature_fft[label + 'peaks_amplitude_std'] = np.std(ampls)

        feature_fft[label + 'peaks_amplitude_min'] = np.min(ampls)
        feature_fft[label + 'peaks_amplitude_max'] = np.max(ampls)
        
        feature_fft[label + 'dominant_amplitude'] = ampls[0]
    else:
        feature_fft[label + 'peaks_amplitude_mean'] = 0
        feature_fft[label + 'peaks_amplitude_std'] = 0
        
        feature_fft[label + 'peaks_amplitude_min'] = 0
        feature_fft[label + 'peaks_amplitude_max'] = 0
        
        feature_fft[label + 'dominant_amplitude'] = 0
    return feature_fft
            
def calculate_fft_feature(time, data, label, threshold = 3):
    '''
    Calculate fourier features:
    Input:
    time - array
    data - pandas.Series or array
    threshold - Hz, minimum detectable peak. detect peaks less and more than threshold
    
    Output:
    fft_feature - dct of features
    
    '''
    feature_fft = {}
    timestamp = (time[len(time)-1] - time[0])/len(time)
    fft_data = np.abs(np.fft.fft(data))
    fft_freq = np.fft.fftfreq(len(fft_data), timestamp)
    peaks, _ = signal.find_peaks(fft_data, fft_data.mean() + 2*np.std(fft_data))## ??? n * std
    labels = np.argsort(fft_freq)
    peaks_label = np.argsort(np.abs(fft_freq[peaks]))
    # add frequency more than 3hz (threshold)
    freqs = np.abs(fft_freq[peaks])[peaks_label] [np.abs(fft_freq[peaks])[peaks_label] > threshold]
    ampls = np.abs(fft_data[peaks])[peaks_label] [np.abs(fft_freq[peaks])[peaks_label] > threshold]
    feature_fft = add_fourier_features(freqs, ampls, feature_fft, label = label + 'more3hz_')
    
    # add frequency less than 3hz (threshold)
    freqs = np.abs(fft_freq[peaks])[peaks_label] [np.abs(fft_freq[peaks])[peaks_label] < threshold]
    ampls = np.abs(fft_data[peaks])[peaks_label] [np.abs(fft_freq[peaks])[peaks_label] < threshold]
    feature_fft = add_fourier_features(freqs, ampls, feature_fft, label = label + 'less3hz_')
    
    feature_fft[label + 'spectrum_energy_mean'] = np.mean(fft_data**2)
    feature_fft[label + 'spectrum_energy_std'] = np.std(fft_data**2)
    return feature_fft

def calculate_batch(acc, hyr, mag, time, patient_info, name, dataframe, exer_keys, index, win_size = 2000, overlap = 0):
    '''
    create batches with win_size and overlap:
    input:
    acc, hyr, mag - data of sensor
    time - array
    overlap - intersection in percents 
    
    '''
    step_size = int(win_size - 0.01*overlap*win_size)
    Data = pd.DataFrame()
    if len(time.values) < win_size:
        return Data
    else:
        count = 0
        for ind in range(0, len(time.values), step_size):
            if ind + win_size < len(time.values):
                df = pd.DataFrame()
                count += 1
                for key in acc.keys():
                    df[key] = [acc[key].values[ind: ind + win_size]]
                
                for key in hyr.keys():
                    df[key] = [hyr[key].values[ind: ind + win_size]]
                    
                for key in mag.keys():
                    df[key] = [mag[key].values[ind: ind + win_size]]
                df['time'] = [time.values[ind: ind + win_size]]
                df['name'] = name
                if patient_info['Степень Паркинсона'][index[0] - 1] in [-1, 'здоров', 'здоровый', 'здоровая']:
                    df['target'] = 0
                else:
                    df['target'] = patient_info['Степень Паркинсона'][index[0] - 1]
                df['exercise_index'] = exer_keys
                Data = Data.append(df, ignore_index=True)
            else:
                break
        return Data
           
def feature_extract(file, calculate_fourier_feature = True):
    features = pd.DataFrame()
    keys = columns_acc + columns_hyr + columns_mag
    for i in file.index:
        feature_one_sensor = {}
        for key in keys:
            feature_one_sensor.update(calculate_feature(pd.DataFrame(file['time'][i]),
                                                      pd.DataFrame(file[key][i]),
                                                      label = key, calculate_fourier_feature= calculate_fourier_feature)
                                     )
        df = pd.DataFrame(feature_one_sensor)
        df['name'] = file['name'][i]
        features = features.append(df, ignore_index= True)
        if i % 100 == 99:
            print(i+1)
    
    features['target'] = file['target'].map({0 : 0, 'П(2)' : 2, 'П(3)' : 3, 'тремор' : 4, 'П(1)' : 1})
    features['exercise_index'] = file['exercise_index']
    return features    

    
def predict(train_features, test_features, classifiers, 
            names_of_features = None, return_feature_importance = False, 
            balanced_accuracy = True, use_dict = False):
    
    '''
    Input
    train_feature, test_features - pandas dataframe with columns "exercise_index" and "target". The target column has values from 0 to 4. Where 0 is health, 1-3 parkinson stage, 4 is different tremor.
    Classifiers - dict of classifiers {'RF' : RandomForestClassifier}
    names_of_features - dict or list of features names.
    Output
    results
    '''
    results = pd.DataFrame()
    

    for i in test_features['exercise_index'].unique():
#         if i == 10:
#             continue
#         if parkinson_stage:
#             X_train = train_features[train_features['target'] != 4]
#             X_test  = test_features [test_features ['target'] != 4]
#         else:
#             X_train = train_features[ (train_features['target'] == 4) + (train_features['target'] == 0) ]
#             X_test  = test_features [ (test_features ['target'] == 4) + (test_features ['target'] == 0) ]
        if use_dict:
            X_train = pd.DataFrame()
            X_test  = pd.DataFrame()

            for name in train_features.name.unique():
                if name not in no_use_this_patients_per_exercise[i]:
                    X_train = X_train.append(train_features[train_features.name == name])
            
            for name in test_features.name.unique():
                if name not in no_use_this_patients_per_exercise[i]:
                    X_test  = X_test.append(test_features[test_features.name == name])
            X_train = shuffler(X_train)
            
        else:
            X_train = train_features
            X_test  = test_features 
        if len(X_test) == 0:
            continue
        if len(X_train.target.unique()) < 2:
            continue
        y_train = X_train[X_train['exercise_index'] == i]['target'].values
        y_test  = X_test [X_test ['exercise_index'] == i]['target'].values
    
        X_train = X_train[X_train['exercise_index'] == i].drop(columns= ['target', 'exercise_index', 'name'])
        X_test  = X_test [X_test ['exercise_index'] == i].drop(columns= ['target', 'exercise_index', 'name'])
        
        if names_of_features:
            if names_of_features is list:
                features_list = names_of_features
            else:
                features_list = names_of_features[i]
            
            X_train = X_train[features_list]
            X_test = X_test[features_list]
        
            
        if return_feature_importance:
            columns = X_train.columns
        X_train = sclr.fit_transform(X_train)
        X_test  = sclr.transform(X_test)
        df = pd.DataFrame()
        df['exer_idx'] = [i]
        df['train_len'] = len(y_train)
        df['test_len'] = len(y_test)
        for key in classifiers:
            model = classifiers[key]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if balanced_accuracy:
                df[key] = [balanced_accuracy_score(y_pred, y_test)]
            else:
                df[key] = [accuracy_score(y_pred, y_test)]

            if (key == 'RF') and (return_feature_importance):
                feature_importance_dict_rf[i] = {}
                feature_importance_dict_rf[i] = pd.Series(classifiers['RF'].feature_importances_,
                                                          index = columns).sort_values(ascending = False)
        results = results.append(df, ignore_index= True)
    if return_feature_importance:
        return results, feature_importance_dict_rf
    else:
        return results

def shuffler(df):
    # return the pandas dataframe
    return df.reindex(np.random.permutation(df.index))
    
def plot_mean_and_variance(X, time, l = 1000, N = 10):
    x_filt = X.rolling(window=N + 1)
    x_filt_mean = x_filt.mean()
    std_h = (X - x_filt_mean)[N:]
    std = np.std(std_h)
    plt.figure(0, figsize=(10,5))
    plt.plot(time[:l], x_filt_mean[N:N+l], c = 'r', label = 'Smooth Signal')
    plt.fill_between(time[:l], x_filt_mean[N:N+l] - 2*std, 
                     x_filt_mean[N:N+l] + 2*std, 
                     color = 'grey', alpha = 0.2, 
                     label = '95% CI, $\sigma$ =' + str(round(std)))
    plt.plot(time[:l], X[:l], c = 'b', label = 'Real Signal', alpha = 0.4)
