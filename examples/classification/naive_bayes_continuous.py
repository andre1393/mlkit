import pandas as pd

from mlkit.classification.naive_bayes import NaiveBayes


def example_tennis_continuous():
    df = pd.read_csv('../datasets/play_tennis_continuous.csv')
    clf = NaiveBayes()
    test = pd.DataFrame({'Outlook': ['Overcast'], 'Temperature': [66], 'Humidity': [90], 'Wind': ['Strong']})
    clf.fit(df.drop('target', axis=1), df['target'])
    print(f'predict: {clf.predict(test)}')
    print('-----------------------------')
    yes_idx = clf.classes_.index('Yes')
    print(f'predict_proba (Yes): {clf.predict_proba(test).apply(lambda x: x[yes_idx])}')


if __name__ == '__main__':
    example_tennis_continuous()
