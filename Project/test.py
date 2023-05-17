import numpy as np
import pickle
import cv2
import random
import matplotlib.pyplot as plt

def decrypt_pickle(file_path):
    with open(file_path, 'rb') as file:
        encrypted_data = file.read()

    # Add your decryption logic here
    decrypted_data = encrypted_data  # Modify this line with your decryption code

    unpickled_data = pickle.loads(decrypted_data)

    return unpickled_data

all_data = decrypt_pickle('D:\ML\catvsnotcat_small.pkl')

IMG_SIZE = 64
label_dict = {'cat': 1, 'not-cat': 0}
all_data_processed = []

number_of_examples = len(all_data)
let_know = int(number_of_examples / 10)

for idx, example in enumerate(all_data):
    if (idx+1)%let_know == 0:
        print(f'processing {idx + 1}')
    resized_down = cv2.resize(example['X'], (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_LINEAR)
    
    all_data_processed.append({'X': np.array(resized_down), 'Y': label_dict[example['Y']]})

classNames = {value:key for key, value in label_dict.items()}
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(18,12))
axes = axes.flatten()
number_of_examples = len(all_data_processed)
for idx, axis in enumerate(axes):
    idx = random.randint(0, number_of_examples)
    example = all_data_processed[idx]
    axis.axis('off')
    axis.set_title(f"{classNames[example['Y']]}")
    axis.imshow(example['X'])

random.seed(42)
X = np.array([example['X'] for example in all_data_processed])
Y = np.array([example['Y'] for example in all_data_processed])
print(f"Rozmiar cech (X): {X.shape}, rozmiar flagi/indykatora klasy (Y): {Y.shape}")
split_ratio = 0.6
split_idx = int(len(Y) * split_ratio)

# One-hot encode Y
num_classes = len(np.unique(Y))
Y_encoded = np.eye(num_classes)[Y.flatten()]

# Split X and Y into train and test sets
X_train = X[:split_idx]
X_test = X[split_idx:]
Y_train = Y_encoded[:split_idx]
Y_test = Y_encoded[split_idx:]

# Flatten X_train and X_test
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Convert Y_train and Y_test to dense matrices
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

print("X_train_flat shape: " + str(X_train_flat.shape))
print("X_test_flat shape: " + str(X_test_flat.shape))
print("Y_train shape: " + str(Y_train.shape))
print("Y_test shape: " + str(Y_test.shape))
