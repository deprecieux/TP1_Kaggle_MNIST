# TP1 Digit Recognizer Déprécieux

# import statements
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    # importing the datasets
    train_df = pd.read_csv('C:/Users/Elitebook HP/Desktop/Digit_Recognizer_Deprecieux/Data/train.csv')
    test_df = pd.read_csv('C:/Users/Elitebook HP/Desktop/Digit_Recognizer_Deprecieux/Data/test.csv')
    
    # prepare training dataset
    imageIds = list(range(1,28001))
    X = train_df.drop('label', axis=1) # features
    Y = train_df['label'] # supervised answer

    # split dataset into training and testing dataset
    # 70% training and 30% testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)
   
    # use the training sets for random forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, Y_train)  
    
    # classify the images in the test dataset
    classified = clf.predict(test_df)
    
    # create submission file for the Kaggle competition with the predictors answers
    df = pd.DataFrame()
    df['ImageId'] = imageIds
    df['Label'] = classified
    df.to_csv('C:/Users/Elitebook HP/Desktop/Digit_Recognizer_Deprecieux/Data/submission.csv', index=False)
