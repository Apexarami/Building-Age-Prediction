**Overview**
This project predicts the age category of buildings based on images using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on a dataset of house images labeled by their construction period.

**Dataset**
The dataset consists of images of buildings categorized by their construction years.
Images are stored in a directory structure where each folder represents an age range (e.g., 1950-1960, 1990-2000).
The dataset is split into training (80%) and validation (20%) subsets.

**Model Architecture**
Input: Images of size 180x180 pixels.
Layers:
1. Rescaling layer (Normalization)
2. 3 Convolutional layers with ReLU activation and max pooling
3. Flatten layer
4. Dense layer (128 neurons, ReLU activation)
5. Output layer (Softmax activation with the number of classes)

**Installation & Dependencies**
To run this project, install the required dependencies:
pip install tensorflow numpy matplotlib
How to Run
1. Prepare the dataset: Place images inside a folder named Houses, where each subfolder represents a building age category.
2. Train the model:
python train.py
3. Make Predictions:
python predict.py --image path/to/image.jpg

**Results**
The model outputs the predicted age category of the building along with the confidence score.
Example prediction:
Predicted class: 1991-2000 (95.20% confidence)
