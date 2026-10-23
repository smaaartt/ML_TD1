from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from churnlab.data import TARGET, encode_target, load_raw, parse_inputs, split_data

df = parse_inputs(load_raw())
train, valid, test = split_data(df) 

first_features = ["tenure", "MonthlyCharges"]
X_train, y_train = train[first_features], encode_target(train[TARGET])
X_valid, y_valid = valid[first_features], encode_target(valid[TARGET])

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

for name, clf in [("baseline", baseline), ("logistic", model)]:
    pred = clf.predict(X_valid)
    print(name, "accuracy =", round(accuracy_score(y_valid, pred), 4))
    print("labels [0=No, 1=Yes], lignes = réel, colonnes = prédit")
    print(confusion_matrix(y_valid, pred, labels=[0, 1]))