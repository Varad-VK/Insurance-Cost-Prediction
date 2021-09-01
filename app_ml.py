# for building the ML model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_model():
    # load the data
    df = pd.read_csv('./insurance.csv')
    print("DATASET SIZE:",df.size)
    print("Dataset Shape:",df.shape)
    print("__________________________________________________________")
    print(df.describe())
    print("__________________________________________________________")
    # clean the data
    # print(df.isna().sum())

    # convert the categorical to numeric
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df['sex'] = encoder.fit_transform(df['sex'])
    df['smoker'] = encoder.fit_transform(df['smoker'])
    df['region'] = encoder.fit_transform(df['region'])
    # print(df.head(20))
    # print(df['region'].unique())

    # sex: 0: female, 1: male
    # smoker: 1: yes, 0: no
    # region: 3: southwest, 2: southeast, 1: northwest, 0: northeast

    # find x and y
    x = df.drop('charges', axis=1) #INDEPENDENT VARIABLE
    y = df['charges']  #Y depend on X

    # split the data into train and test
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=0)
    #changes in training size
    #SIZE
    print("x_train size:",x_train.size)
    print("y_train size:",y_train.size)
    print("x_test size:",x_test.size)
    print("y_test size:",y_test.size)

    # model training
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)

    # dump the model
    import pickle
    file = open('model_lr.pkl', 'wb')
    pickle.dump(model, file)
    file.close()

    def test_model():
        from sklearn.metrics import accuracy_score
        predictions = model.predict(x_test)
        print(predictions)
        print(f"accuracy = {model.score(x_test, y_test)}")
        return  predictions
    predictions = test_model()

    # def create_regression_line():
    #     predictions = model.predict(x)
    #     plt.scatter(x['bmi'], y)
    #     plt.plot(x['bmi'], predictions, color="red")
    #
    #     plt.xlabel('Age')
    #     plt.ylabel('charges')
    #     plt.title("regression line for insurance")
    #     plt.tight_layout()
    #     plt.savefig('static/regression.png')
    #     plt.show()
    #
    # create_regression_line()



def predict(age, sex, bmi, children, smoker, region, algorithm):

    # load the model
    import pickle

    filename = ''
    if algorithm == 0:
        filename = 'model_lr.pkl'

    with open(filename, 'rb') as file:
        model = pickle.load(file)

    charges = model.predict([[age, sex, bmi, children, smoker, region]])

    return charges[0]



if __name__ == '__main__':
    train_model()

    # predict(19, 0, 27.9, 0, 1, 3)
