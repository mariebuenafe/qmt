import pickle
from PIL import Image
from io import BytesIO
from img2vec_pytorch import Img2Vec
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from nltk.classify import NaiveBayesClassifier
import nltk

# Function to download NLTK data
def download_nltk_data():
    nltk.download('punkt')

# Download NLTK data
download_nltk_data()

def word_features(words):
    return dict([(word, True) for word in words])

emojis = {
    'happy': 'Happy😊',
    'sad': 'Sad😢',
    'angry': 'Angry😠',
    'excited': 'Excited😃',
    'scared': 'Scared😨',
}

emotions = {
    'happy': ['happy', 'joyful', 'glad', 'delighted', 'cheerful', 'smile', 'laugh', 'celebrate'],
    'sad': ['sad', 'unhappy', 'disappointed', 'miserable', 'cry', 'tear', 'heartbroken', 'grief'],
    'angry': ['angry', 'frustrated', 'irritated', 'mad', 'furious', 'rage', 'annoyed'],
    'excited': ['excited', 'thrilled', 'eager', 'enthusiastic', 'pumped', 'ecstatic', 'animated'],
    'scared': ['scared', 'fearful', 'anxious', 'nervous', 'terrified', 'panicked', 'worried', 'frightened'],
}

emotion_features = [(word_features(emotion_word.split()), emotion) for emotion, words in emotions.items() for emotion_word in words]
classifier = NaiveBayesClassifier.train(emotion_features)

# Load the trained image classification model
with open('cat_breeds_model.pkl', 'rb') as f:
    img_model = pickle.load(f)

img2vec = Img2Vec()

# Load the dataset
data = pd.read_csv("covid19globalstatisticsdataset.csv")

# Preprocess columns containing non-numeric values
def remove_commas_and_convert_to_numeric(value):
    if isinstance(value, str):
        return float(value.replace(',', ''))
    return value

numerical_cols = ['Total Cases', 'New Cases', 'Total Deaths', 'New Deaths', 'Total Recovered',
                  'New Recovered', 'Active Cases', 'Serious, Critical', 'Tot Cases/1M pop',
                  'Deaths/1M pop', 'Total Tests', 'Tests/1M pop', 'Population']
data[numerical_cols] = data[numerical_cols].applymap(remove_commas_and_convert_to_numeric)

# Fill NaN values with zeros
data.fillna(0, inplace=True)

# Define threshold and create label column
threshold = 1000
data['Country'] = (data['Total Cases'] > threshold).astype(int)

# Define features and target variable
X = data[['Total Cases', 'Total Deaths', 'Total Recovered']]
y = data['Country']

# Initialize and train the SVM model
svm = SVC(kernel='linear')
svm.fit(X, y)

# Scale the features for SVM
scaler = StandardScaler().fit(X)

# Streamlit App
st.set_page_config(layout="wide", page_title="Machine Learning Applications")

# Custom styles for buttons, background, and sidebar
button_style = """
    <style>
        body {
            background-image: url('https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-025.jpg');
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            background-color: #f0f0f0; /* Fallback color if image is not available */
        }
        .css-18e3th9 {
            padding-top: 0;
        }
        .css-1d391kg {
            padding-top: 0;
        }
        .css-1v3fvcr {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 15px;
            padding: 20px;
            margin-top: 20px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 15px 32px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border: none;
            border-radius: 12px;
        }

        /* Styles for sidebar navigation */
        .css-r6a2li {
            background-color: rgba(255, 255, 255, 0.7);
            border-radius: 10px;
            padding: 10px;
            margin-top: 20px;
        }
    </style>
"""

# Display the button style
st.markdown(button_style, unsafe_allow_html=True)


st.sidebar.title("ITEQMT Machine Learning Application Portfolio")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "Prediction", "Sentiment Analysis", "Image Classification", "Sample Source Code"])

if app_mode == "Home":
    st.title("Machine Learning Application")
    st.image("https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-025.jpg", use_column_width=True)
    st.write("Welcome to the Machine Learning App. Use the sidebar to navigate between different functionalities. There, I have 3 applications for you to try.")

    st.header("About Me")
    st.write("I am Perlyn Marie M. Buenafe, currently in my third year at Carlos Hilado Memorial State University pursuing a Bachelor of Science in Information Systems. My enthusiasm lies in the realm of machine learning, where I am dedicated to crafting inventive applications through the utilization of state-of-the-art technologies.")

    st.header("Lesson Learned in Class")
    st.write("""
        During our lessons, I've been exposed to diverse machine learning methodologies and software, encompassing tasks like data preparation, model construction, assessment, and implementation through platforms like Streamlit.
        At the outset of the course, I immersed myself in various exercises to grasp data manipulation, refining, analysis, and model training.
        Moreover, I've acquired practical skills in crafting complete machine learning solutions to address practical challenges.
    """)

    st.header("App Description")
    st.write("""
        This app provides various functionalities such as prediction, sentiment analysis, and image classification using machine learning models.

        **• Prediction**: Make predictions based on input features using a pre-trained machine learning model. This app will predict the country that has a user-based input.

        **• Sentiment Analysis**: Analyze the sentiment of a given text using a pre-trained sentiment analysis model. This app will help the user identify their sentiment based on the sentence they input.

        **• Image Classification**: Categorize images into different classes using a pre-trained image classification model. This app will determine the breed of a cat when the user uploads its image.
        """)

elif app_mode == "Prediction":
    st.title("Prediction Model")
    st.write("Enter details for prediction:")
    total_cases = st.number_input("Total Cases")
    total_deaths = st.number_input("Total Deaths")
    total_recovered = st.number_input("Total Recovered")
    if st.button("Predict Country"):
        # Scale the input features
        scaled_input = scaler.transform([[total_cases, total_deaths, total_recovered]])
        # Predict using the SVM model
        country_prediction = svm.predict(scaled_input)
        # Define the list of country names
        countries = ["USA", "India", "France"]  # Ensure this list matches the order of the model's output
        # Retrieve the predicted country name
        predicted_country = countries[country_prediction[0]]
        st.write(f"Prediction: {predicted_country}")

        # Optional: Evaluate the model performance
        y_pred = svm.predict(X)
        accuracy = np.mean(y_pred == y)
        st.write(f"Model Accuracy: {accuracy:.2%}")

elif app_mode == "Sentiment Analysis":
    st.title("Sentiment Analysis with Emojis")
    sentence = st.text_input("Enter a sentence", "")
    if st.button("Analyze Sentiment"):
        sentiment = classifier.classify(word_features(sentence.split()))
        emoji = emojis.get(sentiment, '❓')
        st.write("Emoji for sentiment:", emoji)

elif app_mode == "Image Classification":
    st.title("Image Classification for Cat Breeds")
    st.write("Upload an image of a cat to classify its breed:")
    my_upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if my_upload is not None:
        image = Image.open(my_upload).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        img_features = img2vec.get_vec(image)
        pred = img_model.predict([img_features])
        st.write(f"Predicted Breed: {pred[0]}")

elif app_mode == "Sample Source Code":
    st.title("Sample Source Code")
    st.write("Here you can find the sample source code for different models used in this application.")

    # Input your Prediction Code here
    if st.button("Prediction Code"):
        # Display the Prediction Code
        st.write("Prediction Code")
        st.code("""
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        import numpy as np

        # Load the dataset
        data = pd.read_csv("covid19globalstatisticsdataset.csv")

        # Define a function to remove commas and convert to numeric
        def remove_commas_and_convert_to_numeric(value):
            if isinstance(value, str):
                return float(value.replace(',', ''))
            return value

        # Preprocess columns containing non-numeric values
        numerical_cols = ['Total Cases', 'New Cases', 'Total Deaths', 'New Deaths', 'Total Recovered',
                          'New Recovered', 'Active Cases', 'Serious, Critical', 'Tot Cases/1M pop',
                          'Deaths/1M pop', 'Total Tests', 'Tests/1M pop', 'Population']
        datasetCSV[numerical_cols] = datasetCSV[numerical_cols].applymap(remove_commas_and_convert_to_numeric)

        # Fill NaN values with zeros
        datasetCSV.fillna(0, inplace=True)

        # Define threshold and create label column
        threshold = 1000
        datasetCSV['Country'] = (datasetCSV['Total Cases'] > threshold).astype(int)

        # Define features and target variable
        X = data[['Total Cases', 'Total Deaths', 'Total Recovered']]
        y = data['Country']

        # Split the data into training and testing sets for Random Forest
        X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

        # Define parameter grid for Random Forest
        param_grid_rf = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Initialize the Random Forest classifier
        rf = RandomForestClassifier()

        # Initialize GridSearchCV
        grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
        # Fit the GridSearchCV to find the best parameters
        grid_search_rf.fit(X_train_rf, y_train_rf)

        # Get the best parameters for Random Forest
        best_params_rf = grid_search_rf.best_params_
        print("Best Parameters for Random Forest:", best_params_rf)

        # Evaluate the Random Forest model
        accuracy_rf = grid_search_rf.score(X_test_rf, y_test_rf)
        print("Accuracy for Random Forest:", accuracy_rf)

        # Sample data (replace this with your actual data) for SVM
        X_svm = np.array([[20, 40, 30], [30, 60, 40], [40, 50, 20], [50, 70, 60], [60, 60, 50]])
        y_svm = np.array([0, 1, 0, 1, 1])

        # Scale the features for SVM
        scaler = StandardScaler()
        X_scaled_svm = scaler.fit_transform(X_svm)

        # Initialize and train the SVM model
        svm = SVC(kernel='linear')
        svm.fit(X_scaled_svm, y_svm)

        # Evaluate the SVM model on the test set
        y_pred_svm = svm.predict(X_scaled_svm)
        print(f'Accuracy for SVM: {accuracy_score(y_svm, y_pred_svm)}')

        # Define the corrected predict_country function
        def predict_country(Total_Cases, Total_Deaths, Total_Recovered):
            scaled_input = scaler.transform([[Total_Cases, Total_Deaths, Total_Recovered]])
            prediction_index = svm.predict(scaled_input)[0]
            country = countries[prediction_index]
            return country  # Return the predicted country name

        # List of country names in the same order as the labels returned by the model
        countries = ["USA", "India", "France", ...]

# Make prediction with the given values
print("Prediction for Country:", predict_country(111_367_209, 109_053_249, 1_771))
        """)

    # Input your Sentiment Analysis Code here
    if st.button("Sentiment Analysis Code"):
        # Display the Sentiment Analysis Code
        st.write("Sentiment Analysis Code")
        st.code("""
        !pip install streamlit

        import streamlit as st
        import nltk
        from nltk.classify import NaiveBayesClassifier

        nltk.download('punkt')

        def word_features(words):
            return dict([(word, True) for word in words])

        emojis = {
            'happy': 'Happy😊',
            'sad': 'Sad😢',
            'angry': 'Angry😠',
            'excited': 'Excited😃',
            'scared': 'Scared😨',
        }

        emotions = {
            'happy': ['happy', 'joyful', 'glad', 'delighted', 'cheerful', 'smile', 'laugh', 'celebrate'],
            'sad': ['sad', 'unhappy', 'disappointed', 'miserable', 'cry', 'tear', 'heartbroken', 'grief'],
            'angry': ['angry', 'frustrated', 'irritated', 'mad', 'furious', 'rage', 'annoyed'],
            'excited': ['excited', 'thrilled', 'eager', 'enthusiastic', 'pumped', 'ecstatic', 'animated'],
            'scared': ['scared', 'fearful', 'anxious', 'nervous', 'terrified', 'panicked', 'worried', 'frightened'],
        }

        emotion_features = [(word_features(emotion_word.split()), emotion) for emotion, words in emotions.items() for emotion_word in words]

        classifier = NaiveBayesClassifier.train(emotion_features)

        with open("BUENAFE_SentimentAnalyzer_Model_StreamlitApp.py", "w") as file:
            file.write
        def analyze_sentiment(sentence):
            sentence = sentence.strip()
            if sentence:
                sentiment = classifier.classify(word_features(sentence.split()))
                emoji = emojis.get(sentiment, '❓')
                return emoji
            else:
                return "No input sentence provided."

        def main():
            st.title("Sentiment Analysis with Emojis")
            sentence = st.text_input("Enter a sentence", "")
            if st.button("Analyze Sentiment"):
                result_emoji = analyze_sentiment(sentence)
                st.write("Emoji for sentiment:", result_emoji)

        if __name__ == "__main__":
            main()
              """)

    # Input your Image Classification Code here
    if st.button("Image Classification Code"):
        # Display the Image Classification Code
        st.write("Image Classification Code")
        st.code("""
        !pip install img2vec_pytorch
        !pip install torch torchvision
        !pip install scikit-learn==1.4.2

        from google.colab import drive
        drive.mount('/content/drive')

        import os
        import shutil
        import numpy as np
        from sklearn.model_selection import train_test_split
        from img2vec_pytorch import Img2Vec
        from PIL import Image
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score
        import pickle

        from img2vec_pytorch import Img2Vec
        from PIL import Image
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score
        from matplotlib import pyplot as plt

        # Define paths
        data_dir = '/content/drive/MyDrive/cat-breeds/cat-breeds'
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')

        # Initialize Img2Vec
        img2vec = Img2Vec()

        # Create train and val directories if they do not exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Function to check if a file is an image
        def is_image_file(filename):
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
            return filename.lower().endswith(valid_extensions)

        # Split the data into training and validation sets
        for category in os.listdir(data_dir):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path) and category not in ['train', 'val']:
                images = [f for f in os.listdir(category_path) if is_image_file(f)]
                if len(images) == 0:
                    print(f"Skipping empty directory: {category_path}")
                    continue
                train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

                # Move training images
                train_category_dir = os.path.join(train_dir, category)
                os.makedirs(train_category_dir, exist_ok=True)
                for img in train_images:
                    shutil.move(os.path.join(category_path, img), os.path.join(train_category_dir, img))

                # Move validation images
                val_category_dir = os.path.join(val_dir, category)
                os.makedirs(val_category_dir, exist_ok=True)
                for img in val_images:
                    shutil.move(os.path.join(category_path, img), os.path.join(val_category_dir, img))

        # Verify directories
        if not os.path.exists(train_dir) or not os.path.exists(val_dir):
            raise FileNotFoundError(f"Check the directory paths. Train: {train_dir}, Val: {val_dir}")

        # Initialize Img2Vec
        img2vec = Img2Vec()

        # Prepare data
        data = {}
        for j, dir_ in enumerate([train_dir, val_dir]):
            features = []
            labels = []
            for category in os.listdir(dir_):
                category_path = os.path.join(dir_, category)
                if os.path.isdir(category_path):
                    for img_path in os.listdir(category_path):
                        img_path_ = os.path.join(category_path, img_path)
                        if is_image_file(img_path_):
                            img = Image.open(img_path_).convert('RGB')
                            img_features = img2vec.get_vec(img)
                            features.append(img_features)
                            labels.append(category)
            data[['training_data', 'validation_data'][j]] = features
            data[['training_labels', 'validation_labels'][j]] = labels

        # Plot sample images
        class_names = sorted(os.listdir(train_dir))
        nrows = len(class_names)
        ncols = 10
        plt.figure(figsize=(ncols*1.5, nrows*1.5))
        for row in range(nrows):
            class_name = class_names[row]
            img_paths = [os.path.join(train_dir, class_name, filename)
                for filename in os.listdir(os.path.join(train_dir, class_name)) if is_image_file(filename)]
            for col in range(min(ncols, len(img_paths))):
                plt.subplot(nrows, ncols, row*ncols + col + 1)
                img = plt.imread(img_paths[col])
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.title(class_name, fontsize=8)
        plt.tight_layout()
        plt.show()

        # Define the parameter grid for SVM
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': [1, 0.1, 0.01, 0.001],
            'kernel': ['rbf', 'poly']
        }

        # Create a GridSearchCV object
        model = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, scoring='accuracy')

        # Fit the model
        model.fit(data['training_data'], data['training_labels'])

        # Print best parameters and score
        print("Best Parameters:", model.best_params_)
        print("Best Score:", model.best_score_)

        # Save the model
        model_path = '/content/drive/MyDrive/cat-breeds/cat_breeds_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Load and test the model (Example)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        img2vec = Img2Vec()
        class_labels = sorted(os.listdir(train_dir))

        image_path = '/content/drive/MyDrive/cat-breeds/test/test_images/test1.jpg'
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Test image not found: {image_path}")

        img = Image.open(image_path).convert('RGB')
        features = img2vec.get_vec(img)
        features_2d = features.reshape(1, -1)

        # Get prediction probabilities
        prediction_probabilities = model.predict_proba(features_2d)[0]
        for ind, prob in enumerate(prediction_probabilities):
            print(f'Class {class_labels[ind]}: {prob*100:.2f}%')

        # Get prediction
        pred = model.predict(features_2d)
        print(f'Predicted Class: {pred[0]}')

        !pip install streamlit
        !pip install img2vec_pytorch

        #note we need to install a specific version to avoid having issues with scikit-learn library
        !pip install scikit-learn==1.4.2


        #Since using StreamLit library requires a python file (.py), this codes writes a python file in Google Colab
        %%writefile app.py
        import pickle
        from PIL import Image
        from io import BytesIO
        from img2vec_pytorch import Img2Vec
        import streamlit as st

        #NOTE don't forget to upload the picke (model) file to your Google Colab First
        #to run this code
        #you can use any model that is capable of classifiying images that uses img2vec_pytorch
        with open('cat_breeds_model.pkl', 'rb') as f:
            model = pickle.load(f)

        img2vec = Img2Vec()

        ## Streamlit Web App Interface
        st.set_page_config(layout="wide", page_title="Image Classification for Cat Breeds")

        st.write("## Image Classification Model in Python!")
        st.write(
            ":grin: Upload an image of a cat. We will try to classify its breed on our trained model! :grin:"
        )
        st.sidebar.write("## Upload and download :gear:")

        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

        # Download the fixed image
        @st.cache_data
        def convert_image(img):
            buf = BytesIO()
            img.save(buf, format="jpg")
            byte_im = buf.getvalue()
            return byte_im

        def fix_image(upload):
            image = Image.open(upload)
            col1.write("Image to be predicted :camera:")
            col1.image(image)

            col2.write("Category :wrench:")
            img = Image.open(my_upload)
            features = img2vec.get_vec(img)
            pred = model.predict([features])

            # print(pred)
            col2.header(pred)
            # st.sidebar.markdown("\n")
            # st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")


        col1, col2 = st.columns(2)
        my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if my_upload is not None:
            if my_upload.size > MAX_FILE_SIZE:
                st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
            else:
                fix_image(upload=my_upload)
        else:
            st.write("Trained by: Perlyn Marie M. Buenafe...")
            # fix_image("./cat.jpg")

        ! wget -q -O - ipv4.icanhazip.com

        ! streamlit run app.py & npx localtunnel --port 8501
  """)
