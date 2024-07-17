# AIDI2004_Lab_4
The Fish Market Predictor project aims to create a machine learning model that predicts the species of fish based on several physical measurements. The model is deployed using Flask and hosted on PythonAnywhere, allowing users to input data through a web interface and receive real-time predictions.

Algorithm Used: 
The project employs a RandomForestClassifier from the scikit-learn library. This algorithm is well-suited for classification tasks and works by building multiple decision trees during training and outputting the mode of the classes (classification) of the individual trees.

Features Used: 
The model uses the following features from the dataset: Length1, Length2, Length3, Height, and Width.

Frontend:
Built using HTML and CSS, the frontend allows users to input fish measurements and see the predicted species.
The web page is styled with CSS, including a background image of fish for a visually appealing interface.

Deployment:
The application is deployed on PythonAnywhere, making it accessible online.
The deployment process includes configuring the web app, installing dependencies, and ensuring the Flask application runs smoothly on the platform.
