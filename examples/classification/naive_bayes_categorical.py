import pandas as pd

from mlkit.classification.naive_bayes import NaiveBayes


def example_tennis_categorical():
    df = pd.read_csv('examples/datasets/play_tennis.csv')
    clf = NaiveBayes()
    test = pd.DataFrame({'Outlook': ['Sunny'], 'Temperature': ['Hot'], 'Humidity': ['Normal'], 'Wind': ['Weak']})
    clf.fit(df.drop('target', axis=1), df['target'])
    print(f'predict: {clf.predict(test)}')
    print('-----------------------------')
    yes_idx = clf.classes_.index('Yes')
    print(f'predict_proba (Yes): {clf.predict_proba(test).apply(lambda x: x[yes_idx])}')


if __name__ == '__main__':
    example_tennis_categorical()
