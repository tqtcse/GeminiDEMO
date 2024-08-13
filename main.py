import librosa
import google.generativeai as genai
import numpy as np
from AI import Model_AI
import pandas as pd


df = pd.read_excel('data1.xlsx')
training_data = []
# print(df.iterrows())
for index, row in df.iterrows():
    # print(index)
    training_data.append({
        'text_input': row['input'],
        'output': row['output'],
    })
# print(training_data)
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr) #tốc độ
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr) #là một đặc trưng thể hiện sự phân bố năng lượng của tần số
    rmse = librosa.feature.rms(y=y) #phản ánh cường độ của âm thanh
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr) #trung tâm khối lượng của phổi tần số
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr) #độ rộng của phổ tần số
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr) #tỷ lệ phần trăm nhất định của tổng năng lượng được chứa
    zcr = librosa.feature.zero_crossing_rate(y) #tần suất mà tín hiệu âm thanh cắt trục không
    mfcc = librosa.feature.mfcc(y=y, sr=sr) #hệ số được tính toán từ phổ của tín hiệu âm thanh

    fft_result = np.fft.fft(y) # năng lượng trung bình của tần số
    fft_magnitude = np.abs(fft_result)
    fundamental_freq = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    harmonic_component, percussive_component = librosa.effects.hpss(y) #thành phần hài âm

    features_dict = {
        'tempo': tempo[0],
        'chroma short-time fourier transform': np.mean(chroma_stft),
        'root mean square energy': np.mean(rmse),
        'spectral centroid': np.mean(spec_cent),
        'spectral bandwidth': np.mean(spec_bw),
        'speactral rolloff': np.mean(rolloff),
        'zero crossing rate': np.mean(zcr),
        'mean fast fourier transform magnitude': np.mean(fft_magnitude),
        'standard fast fourier transform magnitude': np.std(fft_magnitude),
        'mean fundamental frequency': np.mean(fundamental_freq),
        'standard deviation fundamental frequency': np.std(fundamental_freq),
        'standard deviation harmonic component': np.std(harmonic_component),
        'mean harmonic component': np.mean(harmonic_component),
        'mean percussive component': np.mean(percussive_component),
        'standard deviation percussive component': np.std(percussive_component),
    }

    for i, e in enumerate(mfcc):
        features_dict[f'Mel-frequency cepstral coefficients_{i+1}'] = np.mean(e)

    return features_dict

# genai.configure(credentials=load_creds)

file_path = 'Clean4.mp3'
input = str(extract_features(file_path))
# print(str(extract_features(file_path)))
ai = Model_AI()
# ai.printInput("hello")
# ai.create_model("test15data", training_data)
# # ai.create_model("test79","Ai đẹp trai nhất Việt Nam", "Trần Quốc Toàn","Tổng Thống Nga","Putin")
# # ai.call_model("test69")
# # genai.get_model(f'tunedModels/{"test69"}')['status']
model = genai.GenerativeModel(model_name=f'tunedModels/{"test15data"}')
result = model.generate_content("Đây là thông số của loại âm thanh nào?" + input)
print(result.text)
# genai.delete_tuned_model(f'tunedModels/{"test69"}')