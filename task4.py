import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

# Load the dataset
data = pd.read_csv('social_media_reviews.csv')

# Preprocess the dataset
data = data[['review_text', 'sentiment']]
data = data.dropna()
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Split the dataset
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert data to BERT InputExamples
def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(lambda x: InputExample(guid=None, 
                                                             text_a = x[DATA_COLUMN], 
                                                             text_b = None, 
                                                             label = x[LABEL_COLUMN]), axis = 1)

    test_InputExamples = test.apply(lambda x: InputExample(guid=None, 
                                                           text_a = x[DATA_COLUMN], 
                                                           text_b = None, 
                                                           label = x[LABEL_COLUMN]), axis = 1)
    
    return train_InputExamples, test_InputExamples

# Create InputExamples
train_InputExamples, test_InputExamples = convert_data_to_examples(train_data, test_data, 'review_text', 'sentiment')

# Convert InputExamples to InputFeatures
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = [] 
    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length, 
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"], 
                                                     input_dict["token_type_ids"], 
                                                     input_dict["attention_mask"])

        features.append(
            InputFeatures(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          token_type_ids=token_type_ids, 
                          label=e.label)
        )

    def gen():
        for f in features:
            yield ({"input_ids": f.input_ids, 
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids}, f.label)
            
    return tf.data.Dataset.from_generator(gen,
        ({'input_ids': tf.int32, 
          'attention_mask': tf.int32,
          'token_type_ids': tf.int32}, tf.int64),
        ({'input_ids': tf.TensorShape([None]), 
          'attention_mask': tf.TensorShape([None]),
          'token_type_ids': tf.TensorShape([None])}, tf.TensorShape([])))

# Convert to tf.data.Dataset
train_dataset = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)

test_dataset = convert_examples_to_tf_dataset(list(test_InputExamples), tokenizer)
test_dataset = test_dataset.batch(32)

# Build the BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train the model
history = model.fit(train_dataset, epochs=2, validation_data=test_dataset)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

plot_history(history)

# Make predictions
def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_tensors='tf',
        pad_to_max_length=True,
        truncation=True
    )

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    prediction = model.predict([input_ids, attention_mask, token_type_ids])
    sentiment = "Positive" if np.argmax(prediction.logits) == 1 else "Negative"
    score = tf.nn.softmax(prediction.logits).numpy()[0]
    return sentiment, score

# Example prediction
text = "The movie was fantastic and I enjoyed it a lot!"
sentiment, score = predict_sentiment(text)
print(f"Sentiment: {sentiment}, Score: {score}")
