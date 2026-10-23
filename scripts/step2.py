from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from churnlab.data import FEATURES, TARGET, encode_target, load_raw, parse_inputs, split_data
from churnlab.pipeline import build_pipeline

df = parse_inputs(load_raw())
train, valid, test = split_data(df)

X_train, y_train = train[FEATURES], encode_target(train[TARGET])
X_valid, y_valid = valid[FEATURES], encode_target(valid[TARGET])

pipe = build_pipeline(LogisticRegression(max_iter=1000))
pipe.fit(X_train, y_train)         
pred = pipe.predict(X_valid)        

print("accuracy =", round(accuracy_score(y_valid, pred), 4))
print(confusion_matrix(y_valid, pred, labels=[0, 1]))