import os
import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU
from sklearn.model_selection import train_test_split
from pesq import pesq
from pystoi import stoi
import matplotlib.pyplot as plt
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU")
    print("Available devices:", tf.config.list_physical_devices())
else:
    print("TensorFlow is NOT using the GPU. Check your installation.")

def load_data(clean_dir, noisy_dir, sampling_rate=16000):
    clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(".wav")])
    noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith(".wav")])
    clean_signals = [librosa.load(file, sr=sampling_rate)[0] for file in clean_files]
    noisy_signals = [librosa.load(file, sr=sampling_rate)[0] for file in noisy_files]
    return clean_signals, noisy_signals


def augment_data(clean_signals, noise_signals, sampling_rate=16000):
    augmented_clean = []
    augmented_noisy = []
    for clean in clean_signals:
        noise = random.choice(noise_signals)
        noise = np.pad(noise, (0, max(0, len(clean) - len(noise))), mode="constant")
        noise = noise[:len(clean)]
        noisy = clean + 0.5 * noise
        reverb = librosa.effects.preemphasis(noisy)
        augmented_clean.append(clean)
        augmented_noisy.append(reverb)
    return augmented_clean, augmented_noisy

def bucket_batching(clean_signals, noisy_signals, batch_size):
    data = list(zip(clean_signals, noisy_signals))
    data.sort(key=lambda x: len(x[0]))
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return batches

class BucketDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, clean, noisy, batch_size):
        self.batches = bucket_batching(clean, noisy, batch_size)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        clean_batch, noisy_batch = zip(*self.batches[idx])
        noisy_padded = tf.keras.preprocessing.sequence.pad_sequences(noisy_batch, padding="post", dtype="float32")
        clean_padded = tf.keras.preprocessing.sequence.pad_sequences(clean_batch, padding="post", dtype="float32")
        return np.expand_dims(noisy_padded, -1), np.expand_dims(clean_padded, -1)

def conv_tasnet_block(inputs, filters, kernel_size, dilation_rate):
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="causal")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def build_model(input_shape, filters=64, kernel_size=16, num_blocks=8):
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(num_blocks):
        x = conv_tasnet_block(x, filters, kernel_size, 2**i)
    outputs = Conv1D(1, kernel_size=1, activation="linear")(x)
    return tf.keras.Model(inputs, outputs)

clean_dir_train = "/content/unzipped_files/clean_trainset_28spk_wav"
noisy_dir_train = "/content/unzipped_files/noisy_trainset_28spk_wav"
clean_dir_test = "/content/unzipped_files/clean_testset_wav"
noisy_dir_test = "/content/unzipped_files/noisy_testset_wav"


train_clean, train_noisy = load_data(clean_dir_train, noisy_dir_train)
test_clean, test_noisy = load_data(clean_dir_test, noisy_dir_test)
aug_train_clean, aug_train_noisy = augment_data(train_clean, train_noisy)
train_clean.extend(aug_train_clean)
train_noisy.extend(aug_train_noisy)

train_clean, val_clean, train_noisy, val_noisy = train_test_split(train_clean, train_noisy, test_size=0.2, random_state=42)

batch_size = 4
train_gen = BucketDataGenerator(train_clean, train_noisy, batch_size)
val_gen = BucketDataGenerator(val_clean, val_noisy, batch_size)

input_shape = (None, 1)
denoising_model = build_model(input_shape)
dereverberation_model = build_model(input_shape)

inputs = Input(shape=input_shape)
denoised_output = denoising_model(inputs)
dereverberated_output = dereverberation_model(denoised_output)
two_stage_model = tf.keras.Model(inputs, dereverberated_output)
two_stage_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
two_stage_model.summary()

history = two_stage_model.fit(train_gen, validation_data=val_gen, epochs=40, verbose=1)

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

def calculate_metrics(clean_signals, noisy_signals, enhanced_signals, sampling_rate=16000):
    noisy_pesq_scores = [
        pesq(sampling_rate, clean, noisy, 'wb') for clean, noisy in zip(clean_signals, noisy_signals)
    ]
    noisy_stoi_scores = [
        stoi(clean, noisy, sampling_rate) for clean, noisy in zip(clean_signals, noisy_signals)
    ]
    enhanced_pesq_scores = [
        pesq(sampling_rate, clean, enhanced, 'wb') for clean, enhanced in zip(clean_signals, enhanced_signals)
    ]
    enhanced_stoi_scores = [
        stoi(clean, enhanced, sampling_rate) for clean, enhanced in zip(clean_signals, enhanced_signals)
    ]
    return {
        "Noisy PESQ": np.mean(noisy_pesq_scores),
        "Noisy STOI": np.mean(noisy_stoi_scores),
        "Enhanced PESQ": np.mean(enhanced_pesq_scores),
        "Enhanced STOI": np.mean(enhanced_stoi_scores)
    }

def generate_results(model, clean_signals, noisy_signals):
    enhanced_signals = [
        model.predict(np.expand_dims(noisy, axis=-1)).squeeze()
        for noisy in noisy_signals
    ]
    metrics = calculate_metrics(clean_signals, noisy_signals, enhanced_signals)
    return metrics

metrics_test = generate_results(two_stage_model, test_clean, test_noisy)
print("Matched Conditions Metrics:")
print(f"Noisy PESQ: {metrics_test['Noisy PESQ']:.2f}, Enhanced PESQ: {metrics_test['Enhanced PESQ']:.2f}")
print(f"Noisy STOI: {metrics_test['Noisy STOI']:.2f}, Enhanced STOI: {metrics_test['Enhanced STOI']:.2f}")

mismatched_clean, mismatched_noisy = augment_data(test_clean, test_noisy)
metrics_mismatched = generate_results(two_stage_model, mismatched_clean, mismatched_noisy)
print("Mismatched Conditions Metrics:")
print(f"Noisy PESQ: {metrics_mismatched['Noisy PESQ']:.2f}, Enhanced PESQ: {metrics_mismatched['Enhanced PESQ']:.2f}")
print(f"Noisy STOI: {metrics_mismatched['Noisy STOI']:.2f}, Enhanced STOI: {metrics_mismatched['Enhanced STOI']:.2f}")

def create_results_table(metrics_test, metrics_mismatched):
    data = {
        "Condition": ["Matched", "Mismatched"],
        "Noisy PESQ": [metrics_test["Noisy PESQ"], metrics_mismatched["Noisy PESQ"]],
        "Enhanced PESQ": [metrics_test["Enhanced PESQ"], metrics_mismatched["Enhanced PESQ"]],
        "Noisy STOI": [metrics_test["Noisy STOI"], metrics_mismatched["Noisy STOI"]],
        "Enhanced STOI": [metrics_test["Enhanced STOI"], metrics_mismatched["Enhanced STOI"]],
    }
    df = pd.DataFrame(data)
    return df

results_table = create_results_table(metrics_test, metrics_mismatched)
print("\nComparison of STOI and PESQ Values:")
print(results_table)

conditions = ["Matched", "Mismatched"]
noisy_pesq = [metrics_test["Noisy PESQ"], metrics_mismatched["Noisy PESQ"]]
enhanced_pesq = [metrics_test["Enhanced PESQ"], metrics_mismatched["Enhanced PESQ"]]
noisy_stoi = [metrics_test["Noisy STOI"], metrics_mismatched["Noisy STOI"]]
enhanced_stoi = [metrics_test["Enhanced STOI"], metrics_mismatched["Enhanced STOI"]]

plt.figure(figsize=(10, 6))
x = range(len(conditions))
plt.bar(x, noisy_pesq, width=0.4, label='Noisy PESQ', align='center')
plt.bar([p + 0.4 for p in x], enhanced_pesq, width=0.4, label='Enhanced PESQ', align='center')
plt.xticks([p + 0.2 for p in x], conditions)
plt.xlabel('Conditions')
plt.ylabel('PESQ Score')
plt.title('Noisy PESQ vs Enhanced PESQ')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x, noisy_stoi, width=0.4, label='Noisy STOI', align='center')
plt.bar([p + 0.4 for p in x], enhanced_stoi, width=0.4, label='Enhanced STOI', align='center')
plt.xticks([p + 0.2 for p in x], conditions)
plt.xlabel('Conditions')
plt.ylabel('STOI Score')
plt.title('Noisy STOI vs Enhanced STOI')
plt.legend()
plt.show()

"""
import os
import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU
from sklearn.model_selection import train_test_split
from pesq import pesq
from pystoi import stoi
import matplotlib.pyplot as plt
import pandas as pd

def load_data(clean_dir, noisy_dir, sampling_rate=16000):
    clean_files = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith(".wav")])
    noisy_files = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith(".wav")])
    clean_signals = [librosa.load(file, sr=sampling_rate)[0] for file in clean_files]
    noisy_signals = [librosa.load(file, sr=sampling_rate)[0] for file in noisy_files]
    return clean_signals, noisy_signals

def augment_data(clean_signals, noise_signals, sampling_rate=16000):
    augmented_clean = []
    augmented_noisy = []
    for clean in clean_signals:
        noise = random.choice(noise_signals)
        noise = np.pad(noise, (0, max(0, len(clean) - len(noise))), mode="constant")
        noise = noise[:len(clean)]
        noisy = clean + 0.5 * noise
        reverb = librosa.effects.preemphasis(noisy)
        augmented_clean.append(clean)
        augmented_noisy.append(reverb)
    return augmented_clean, augmented_noisy

def bucket_batching(clean_signals, noisy_signals, batch_size):
    data = list(zip(clean_signals, noisy_signals))
    data.sort(key=lambda x: len(x[0]))
    batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
    return batches

class BucketDataGenerator(tf.keras.utils.Sequence):
    def _init_(self, clean, noisy, batch_size):
        self.batches = bucket_batching(clean, noisy, batch_size)

    def _len_(self):
        return len(self.batches)

    def _getitem_(self, idx):
        clean_batch, noisy_batch = zip(*self.batches[idx])
        noisy_padded = tf.keras.preprocessing.sequence.pad_sequences(noisy_batch, padding="post", dtype="float32")
        clean_padded = tf.keras.preprocessing.sequence.pad_sequences(clean_batch, padding="post", dtype="float32")
        return np.expand_dims(noisy_padded, -1), np.expand_dims(clean_padded, -1)

def conv_tasnet_block(inputs, filters, kernel_size, dilation_rate):
    x = Conv1D(filters, kernel_size, dilation_rate=dilation_rate, padding="causal")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def build_model(input_shape, filters=64, kernel_size=16, num_blocks=5):
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(num_blocks):
        x = conv_tasnet_block(x, filters, kernel_size, 2**i)
    outputs = Conv1D(1, kernel_size=1, activation="linear")(x)
    return tf.keras.Model(inputs, outputs)

clean_dir_train = "/content/unzipped_files/clean_trainset_28spk_wav"
noisy_dir_train = "/content/unzipped_files/noisy_trainset_28spk_wav"
clean_dir_test = "/content/unzipped_files/clean_testset_wav/clean_testset_wav"
noisy_dir_test = "/content/unzipped_files/noisy_testset_wav/noisy_testset_wav"

train_clean, train_noisy = load_data(clean_dir_train, noisy_dir_train)
test_clean, test_noisy = load_data(clean_dir_test, noisy_dir_test)
aug_train_clean, aug_train_noisy = augment_data(train_clean, train_noisy)
train_clean.extend(aug_train_clean)
train_noisy.extend(aug_train_noisy)

train_clean, val_clean, train_noisy, val_noisy = train_test_split(train_clean, train_noisy, test_size=0.2, random_state=42)

batch_size = 8
train_gen = BucketDataGenerator(train_clean, train_noisy, batch_size)
val_gen = BucketDataGenerator(val_clean, val_noisy, batch_size)

input_shape = (None, 1)
denoising_model = build_model(input_shape, num_blocks=5)
dereverberation_model = build_model(input_shape, num_blocks=5)

inputs = Input(shape=input_shape)
denoised_output = denoising_model(inputs)
dereverberated_output = dereverberation_model(denoised_output)
two_stage_model = tf.keras.Model(inputs, dereverberated_output)
two_stage_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
two_stage_model.summary()

history = two_stage_model.fit(train_gen, validation_data=val_gen, epochs=1, verbose=1)

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

def calculate_metrics(clean_signals, noisy_signals, enhanced_signals, sampling_rate=16000):
    noisy_pesq_scores = [
        pesq(sampling_rate, clean, noisy, 'wb') for clean, noisy in zip(clean_signals, noisy_signals)
    ]
    noisy_stoi_scores = [
        stoi(clean, noisy, sampling_rate) for clean, noisy in zip(clean_signals, noisy_signals)
    ]
    enhanced_pesq_scores = [
        pesq(sampling_rate, clean, enhanced, 'wb') for clean, enhanced in zip(clean_signals, enhanced_signals)
    ]
    enhanced_stoi_scores = [
        stoi(clean, enhanced, sampling_rate) for clean, enhanced in zip(clean_signals, enhanced_signals)
    ]
    return {
        "Noisy PESQ": np.mean(noisy_pesq_scores),
        "Noisy STOI": np.mean(noisy_stoi_scores),
        "Enhanced PESQ": np.mean(enhanced_pesq_scores),
        "Enhanced STOI": np.mean(enhanced_stoi_scores)
    }

def generate_results(model, clean_signals, noisy_signals):
    enhanced_signals = [
        model.predict(np.expand_dims(noisy, axis=-1)).squeeze()
        for noisy in noisy_signals
    ]
    metrics = calculate_metrics(clean_signals, noisy_signals, enhanced_signals)
    return metrics

metrics_test = generate_results(two_stage_model, test_clean, test_noisy)
print("Matched Conditions Metrics:")
print(f"Noisy PESQ: {metrics_test['Noisy PESQ']:.2f}, Enhanced PESQ: {metrics_test['Enhanced PESQ']:.2f}")
print(f"Noisy STOI: {metrics_test['Noisy STOI']:.2f}, Enhanced STOI: {metrics_test['Enhanced STOI']:.2f}")

mismatched_clean, mismatched_noisy = augment_data(test_clean, test_noisy)
metrics_mismatched = generate_results(two_stage_model, mismatched_clean, mismatched_noisy)
print("Mismatched Conditions Metrics:")
print(f"Noisy PESQ: {metrics_mismatched['Noisy PESQ']:.2f}, Enhanced PESQ: {metrics_mismatched['Enhanced PESQ']:.2f}")
print(f"Noisy STOI: {metrics_mismatched['Noisy STOI']:.2f}, Enhanced STOI: {metrics_mismatched['Enhanced STOI']:.2f}")

def create_results_table(metrics_test, metrics_mismatched):
    data = {
        "Condition": ["Matched", "Mismatched"],
        "Noisy PESQ": [metrics_test["Noisy PESQ"], metrics_mismatched["Noisy PESQ"]],
        "Enhanced PESQ": [metrics_test["Enhanced PESQ"], metrics_mismatched["Enhanced PESQ"]],
        "Noisy STOI": [metrics_test["Noisy STOI"], metrics_mismatched["Noisy STOI"]],
        "Enhanced STOI": [metrics_test["Enhanced STOI"], metrics_mismatched["Enhanced STOI"]],
    }
    df = pd.DataFrame(data)
    return df

results_table = create_results_table(metrics_test, metrics_mismatched)
print("\nComparison of STOI and PESQ Values:")
print(results_table)

conditions = ["Matched", "Mismatched"]
noisy_pesq = [metrics_test["Noisy PESQ"], metrics_mismatched["Noisy PESQ"]]
enhanced_pesq = [metrics_test["Enhanced PESQ"], metrics_mismatched["Enhanced PESQ"]]
noisy_stoi = [metrics_test["Noisy STOI"], metrics_mismatched["Noisy STOI"]]
enhanced_stoi = [metrics_test["Enhanced STOI"], metrics_mismatched["Enhanced STOI"]]

plt.figure(figsize=(10, 6))
x = range(len(conditions))
plt.bar(x, noisy_pesq, width=0.4, label='Noisy PESQ', align='center')
plt.bar([p + 0.4 for p in x], enhanced_pesq, width=0.4, label='Enhanced PESQ', align='center')
plt.xticks([p + 0.2 for p in x], conditions)
plt.xlabel('Conditions')
plt.ylabel('PESQ Score')
plt.title('Noisy PESQ vs Enhanced PESQ')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(x, noisy_stoi, width=0.4, label='Noisy STOI', align='center')
plt.bar([p + 0.4 for p in x], enhanced_stoi, width=0.4, label='Enhanced STOI', align='center')
plt.xticks([p + 0.2 for p in x], conditions)
plt.xlabel('Conditions')
plt.ylabel('STOI Score')
plt.title('Noisy STOI vs Enhanced STOI')
plt.legend()
plt.show()
"""