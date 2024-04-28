# ML_Project4-ImdbSentimentReviewAnalysis


### IMDB Sentiment Review Analysis - Using TensorFlow Hub
This project delves into the fascinating world of sentiment analysis, where we aim to classify movie reviews as positive or negative using machine learning. We leverage the power of TensorFlow and TensorFlow Hub to build a model capable of identifying a reviewer's sentiment towards a film.

### Data Acquisition and Preprocessing:

We employ the tfds.load function from TensorFlow Datasets to access the IMDB movie review dataset. This dataset provides a wealth of text reviews along with their corresponding sentiment labels (positive or negative).
The dataset is thoughtfully split into training, validation, and testing sets using a 60/20/20 ratio. The training set is used to train the model, the validation set is used for hyperparameter tuning, and the testing set is used for final evaluation of the model's performance.


### Text Embedding with TensorFlow Hub:

A crucial step in sentiment analysis involves converting textual data into numerical representations that a machine learning model can understand. This project leverages a pre-trained text embedding layer from TensorFlow Hub.
The specific layer used here (https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1) transforms each review into a 20-dimensional vector, capturing the semantic meaning of the words within the review.


### Model Architecture:

The core of this project is a sequential model built using TensorFlow's keras library.
The model starts with the pre-trained text embedding layer from TensorFlow Hub, which processes the review text and generates a 20-dimensional vector.
This vector is then fed into a dense layer with 16 neurons and a ReLU activation function. This layer helps extract higher-level features from the embedded text representation.
Finally, a single neuron with a sigmoid activation function is used to predict the sentiment of the review. The sigmoid activation outputs a value between 0 and 1, where values closer to 1 indicate a positive sentiment and values closer to 0 indicate a negative sentiment.


### Model Training and Evaluation:

The model is compiled using the Adam optimizer, a binary cross-entropy loss function (suitable for binary classification problems like sentiment analysis), and the accuracy metric to track its performance during training.
We train the model for 20 epochs, where each epoch involves iterating through the entire training dataset once.
After training, the model is evaluated on the unseen testing set to assess its generalization capability. The results (loss and accuracy) are displayed, providing insights into the model's effectiveness.


### Visualization (Optional):
The code includes optional sections for visualizing the model's training process. You can uncomment these sections to generate plots that depict the relationship between epochs and training/validation accuracy or loss. These plots can be helpful in understanding how the model performs over time and identifying potential issues like overfitting.


### Further Exploration:

This project lays a solid foundation for exploring sentiment analysis further. Here are some potential areas for future investigation:
Experimenting with different text embedding layers or techniques.
Adding more layers or changing the model architecture for potentially better performance.
Exploring other sentiment analysis datasets like Amazon product reviews or Twitter sentiment analysis.
Deploying the model as a web application to allow users to analyze their own movie reviews.
