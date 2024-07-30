import numpy as np # type: ignore
import gzip
import matplotlib.pyplot as plt
from urllib import request
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load MNIST data
def load_mnist_data():
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }

    def download_and_extract(url, filename):
        print(f"Downloading {url}{filename}")
        response = request.urlopen(f"{url}{filename}")
        with open(filename, 'wb') as f:
            f.write(response.read())
        with gzip.open(filename, 'rb') as f:
            if 'images' in filename:
                return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
            elif 'labels' in filename:
                return np.frombuffer(f.read(), np.uint8, offset=8)

    data = {}
    for key, value in files.items():
        data[key] = download_and_extract(url_base, value)

    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

train_images, train_labels, test_images, test_labels = load_mnist_data()

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0

# One-hot encode labels
encoder = OneHotEncoder(sparse=False, categories='auto')
train_labels = encoder.fit_transform(train_labels.reshape(-1, 1))
test_labels = encoder.transform(test_labels.reshape(-1, 1))

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# Define neural network
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        m = y.shape[0]
        d_z2 = output - y
        d_W2 = np.dot(self.a1.T, d_z2) / m
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) / m
        d_a1 = np.dot(d_z2, self.W2.T)
        d_z1 = d_a1 * self.sigmoid_derivative(self.a1)
        d_W1 = np.dot(X.T, d_z1) / m
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if (epoch + 1) % 100 == 0:
                loss = self.compute_loss(y, output)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

    def compute_loss(self, y, output):
        m = y.shape[0]
        log_likelihood = -np.log(output[range(m), np.argmax(y, axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Initialize and train the neural network
input_size = 28 * 28
hidden_size = 64
output_size = 10
epochs = 1000
learning_rate = 0.1

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X_train, y_train, epochs, learning_rate)

# Evaluate the neural network
def evaluate(model, X, y):
    predictions = model.predict(X)
    labels = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == labels)
    return accuracy

train_accuracy = evaluate(nn, X_train, y_train)
val_accuracy = evaluate(nn, X_val, y_val)
test_accuracy = evaluate(nn, test_images, test_labels)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot a few predictions
def plot_predictions(model, X, y, num_images=5):
    predictions = model.predict(X)
    labels = np.argmax(y, axis=1)
    for i in range(num_images):
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"True Label: {labels[i]}, Predicted: {predictions[i]}")
        plt.show()

plot_predictions(nn, test_images, test_labels)

