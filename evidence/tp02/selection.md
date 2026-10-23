Selection note: 

Criterion (chosen before the comparison): average precision, computed from  scores of the positive class (Churn = Yes). I chose it because the classes are imbalanced (only 26.5% of customers churn) and because it does not depend on a threshold.

Other metrics (precision, recall and F1 at a 0.5 threshold, accuracy and the confusion matrix) are reported as supporting evidence only.

The decision was made on the validation set only. The test set was not used.

Result: the random forest has the best average precision (0.670), ahead of logistic regression (0.647) and the decision tree (0.603). All three clearly beat the majority-class baseline (0.265).

Decision: I keep the random forest. The gap with logistic regression is small(+0.023) and comes from a single validation split, so I do not claim it is a proven improvement. I keep the forest because it follows the rule I fixed before looking at the results, and its extra cost is negligible here (about 0.5 seconds to train). Recall is still low for all models at the 0.5 threshold (about 50-60%), so a threshold choice would be a useful next step.