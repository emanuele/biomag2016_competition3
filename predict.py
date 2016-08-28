import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegressionCV
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold

from load import create_train_test_sets
from scipy.io import savemat


if __name__ == '__main__':
    print("Biomag2016: Competition 3")
    print("Our attempt uses pyRiemann with a sklearn classifier.")
    subject = 1
    window_size = 150  # temporal window of interst, in timesteps
    t_offset = 15  # beginning of the time window of, from onset
    normalize = False  # unnecessary
    estimator = 'oas'  # covariance estimator
    metric = 'riemann'  # metric for the tangent space
    scoring = 'roc_auc'  # scoring metric
    label = 4  # the label of interest is 4, i.e. "happy"
    cv = 10  # folds for cross-validation

    print("Loading data of subject %d." % subject)
    X_train, y_train, X_test = create_train_test_sets(subject=subject,
                                                      window_size=window_size,
                                                      t_offset=t_offset,
                                                      normalize=normalize)
    y_train = (y_train == label).astype(np.int)  # the labels
    X_all = np.vstack([X_train, X_test])

    print("Estimating covariance matrices with covariance estimator '%s'."
          % estimator)
    cov_all = Covariances(estimator=estimator).fit_transform(X_all)
    print("Computing TangentSpace with metric %s" % metric)
    ts_all = TangentSpace(metric=metric).fit_transform(cov_all)
    ts_train = ts_all[:X_train.shape[0], :]
    ts_test = ts_all[X_train.shape[0]:, :]
    print("Cross validated %s:" % scoring)
    clf = LogisticRegressionCV()
    print("Classifier: %s" % clf)
    cv = StratifiedKFold(y_train, n_folds=cv)
    score = cross_val_score(clf, ts_train, y_train, scoring=scoring,
                            cv=cv, n_jobs=-1)
    print("Label %d, %s = %f" % (label, scoring, score.mean()))

    print("")
    print("Training on training data.")
    clf = LogisticRegressionCV()
    clf.fit(ts_train, y_train)
    print("Predicting test data.")
    y_test = clf.predict_proba(ts_test)
    filename = 'subject%d.mat' % subject
    print("Saving predictions to %s" % filename)
    savemat(file_name=filename,
            mdict={'predicted_probability': y_test[:, 1]})


# This is the output of the code on the competition dataset:
#
# Biomag2016: Competition 3
# Loading data of subject 1.
# Estimating covariance matrices with covariance estimator 'oas'.
# Computing TangentSpace with metric riemann
# Cross validated roc_auc:
# Classifier: LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
#            fit_intercept=True, intercept_scaling=1.0, max_iter=100,
#            multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#            refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)
# Label 4, roc_auc = 0.997500
#
#
# Biomag2016: Competition 3
# Loading data of subject 2.
# Estimating covariance matrices with covariance estimator 'oas'.
# Computing TangentSpace with metric riemann
# Cross validated roc_auc:
# Classifier: LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
#            fit_intercept=True, intercept_scaling=1.0, max_iter=100,
#            multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#            refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)
# Label 4, roc_auc = 0.846250
#
#
# Biomag2016: Competition 3
# Loading data of subject 3.
# Estimating covariance matrices with covariance estimator 'oas'.
# Computing TangentSpace with metric riemann
# Cross validated roc_auc:
# Classifier: LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
#            fit_intercept=True, intercept_scaling=1.0, max_iter=100,
#            multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#            refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)
# Label 4, roc_auc = 0.917500
#
#
# Biomag2016: Competition 3
# Loading data of subject 4.
# Estimating covariance matrices with covariance estimator 'oas'.
# Computing TangentSpace with metric riemann
# Cross validated roc_auc:
# Classifier: LogisticRegressionCV(Cs=10, class_weight=None, cv=None, dual=False,
#            fit_intercept=True, intercept_scaling=1.0, max_iter=100,
#            multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
#            refit=True, scoring=None, solver='lbfgs', tol=0.0001, verbose=0)
# Label 4, roc_auc = 0.976250
