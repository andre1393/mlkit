import pandas as pd

from mlkit.classification.naive_bayes import NaiveBayes


def example_tennis_categorical():
    df = pd.read_csv('../datasets/play_tennis_continuous.csv')
    clf = NaiveBayes()
    test = pd.DataFrame({'Outlook': ['Sunny'], 'Temperature': ['Hot'], 'Humidity': ['Normal'], 'Wind': ['Weak']})
    clf.fit(df.drop('target', axis=1), df['target'])
    print(clf.predict(test))


if __name__ == '__main__':
    example_tennis_categorical()
