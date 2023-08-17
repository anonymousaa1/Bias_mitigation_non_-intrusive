import tensorflow as tf
from transformers import BertConfig, BertTokenizer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from keras.utils import plot_model
from tqdm import tqdm

num_epochs = 3
max_seq_length = 128
batch_size = 32

directory_path = "output/zari-bert-cda"

from transformers import BertConfig, BertModel
from transformers import TFBertModel
from transformers import load_tf_weights_in_bert

# load config
config = BertConfig.from_json_file(directory_path + "/bert_config.json")
# instantiate model from config
bert_model = TFBertModel(config)
# load pretrained weights
path = directory_path + "/model.ckpt.index"
load_tf_weights_in_bert(bert_model, config, path)
# instantiate tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
tokenizer = BertTokenizer(directory_path + '/vocab.txt', do_lower_case=True)

# an example
# inputs = tokenizer("i have all dogs", return_tensors="tf")
inputs = tokenizer("i have all dogs", padding='max_length', truncation=True, max_length=max_seq_length,
                                 return_tensors="tf")
output = bert_model(inputs)
print(output)
pooled_output = output.pooler_output
print("-->pooled_output:", pooled_output)
classifier = tf.keras.layers.Dense(2, activation='softmax')(pooled_output)
print("-->classifier", classifier)

def do_train(bert_model, tokenizer, max_seq_length, num_epochs, batch_size, texts, labels):
    print("-->device:", tf.config.list_physical_devices())

    # Tokenize input texts
    tokenized_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=max_seq_length,
                                 return_tensors="tf")
    print("-->tokenized_inputs", tokenized_inputs)

    # Convert labels to TensorFlow tensors
    labels = tf.constant([[(l + 1) % 2, (l + 2) % 2] for l in labels], dtype=tf.float32)

    # Get the input tensors
    input_ids = tokenized_inputs["input_ids"]
    token_type_ids = tokenized_inputs["token_type_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    input_ids_input = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    token_type_ids_input = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    attention_mask_input = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    # Freeze the weights of the BERT model
    bert_model.trainable = False
    bert_output = bert_model([input_ids_input, token_type_ids_input, attention_mask_input]).pooler_output
    classifier = tf.keras.layers.Dense(2, activation='softmax')(bert_output)

    classification_model = tf.keras.Model(inputs=[input_ids_input, token_type_ids_input, attention_mask_input],
                                          outputs=classifier)
    print("-->classification_model", classification_model)
    plot_model(classification_model, "cda_model.png")

    # Compile the model
    classification_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Train the model
    classification_model.fit([input_ids, token_type_ids, attention_mask], labels, batch_size=batch_size,
                             epochs=num_epochs, verbose=1)
    return classification_model

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
            FP += 1
        if y_actual[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
            FN += 1

    return TP, FP, TN, FN

# evaluation
def do_evaluation(test_texts, label, classification_model):
    predict_labels = []
    for text in tqdm(test_texts):
        test_input = tokenizer(text, padding='max_length', truncation=True, max_length=max_seq_length,
                               return_tensors="tf")
        output = classification_model(test_input)
        labels = [pro.index(max(pro)) for pro in output.numpy().tolist()]
        predict_labels.append(labels[0])
    TP, FP, TN, FN = perf_measure(label, predict_labels)
    training_acc = accuracy_score(label, predict_labels)
    f1 = f1_score(label, predict_labels)
    fpr_train = FP / float(FP + TN)
    fnr_train = FN / float(FN + TP)
    print("-->ACC", training_acc)
    print("-->F1", f1)
    print("-->FPR:", fpr_train)
    print("-->FNR:", fnr_train)

# # training dataset
train_data = pd.read_csv("dataset/hate_speech_white/train.csv")
test_data = pd.read_csv("dataset/hate_speech_white/test.csv")
dataset_train = train_data.sample(frac=1, random_state=999)
dataset_test = test_data.sample(frac=1, random_state=999)
##### training  #####
texts = dataset_train['text'].tolist()
labels = dataset_train['label'].tolist()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print('No GPU found')

with tf.device('/GPU:0'):
    classification_model = do_train(bert_model, tokenizer, max_seq_length, num_epochs, batch_size, texts, labels)

    # test_texts = ['i fuck it all', 'i love you']
    # inputs = tokenizer(test_texts, padding='max_length', truncation=True, max_length=max_seq_length,
    #                        return_tensors="tf")
    # output = classification_model(inputs)
    # print("-->output", output)
    #
    # Save model
    # classification_model.save("output/cda_model_twitter.h5")
    classification_model.save("output/dropout_model_white.h5")

    # Load model
    # input_ids_input = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    # token_type_ids_input = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    # attention_mask_input = tf.keras.Input(shape=(max_seq_length,), dtype=tf.int32)
    # bert_output = bert_model([input_ids_input, token_type_ids_input, attention_mask_input]).pooler_output
    # classifier = tf.keras.layers.Dense(2, activation='softmax')(bert_output)
    # classification_model = tf.keras.Model(inputs=[input_ids_input, token_type_ids_input, attention_mask_input],
    #                                           outputs=classifier)
    # classification_model.load_weights("output/cda_model_twitter.h5")

    test_texts = dataset_test['text'].tolist()
    label = dataset_test['label'].tolist()
    # with original bert and un-trained classifier
    # bert_output = bert_model([input_ids_input, token_type_ids_input, attention_mask_input]).pooler_output
    # classifier = tf.keras.layers.Dense(2, activation='softmax')(bert_output)
    # classification_model = tf.keras.Model(inputs=[input_ids_input, token_type_ids_input, attention_mask_input],
    #                                           outputs=classifier)

    do_evaluation(test_texts, label, classification_model)

