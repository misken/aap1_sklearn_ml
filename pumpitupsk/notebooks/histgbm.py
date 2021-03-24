import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV

# explicitly require this experimental feature
from sklearn.experimental import enable_hist_gradient_boosting
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingClassifier

# Read data
train_X = pd.read_csv('./data/train_x.csv')
train_y = pd.read_csv("./data/raw/train_y.csv")
test_X = pd.read_csv("./data/test_x.csv")

# drop id col in target (id in feature dataframes will get dropped during preprocessing
train_y = train_y.iloc[:, 1]

# Which cols are numeric and categorical
categorical_cols = train_X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = train_X.select_dtypes(include=['number']).columns.tolist()

# Create transformers
numeric_transformer = StandardScaler()
categorical_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)

# Create preprocessor for column transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_cols),
        ('num', numeric_transformer, numeric_cols)], remainder='drop')

# Columns get reordered during preprocessing based on transformer order
# Need this of indexes to pass in to HistGradientBoostingClassifier
categorical_cols_idx = [_ for _ in range(len(categorical_cols))]

# Append classifier to preprocessing pipeline.
hgbc_clf = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', HistGradientBoostingClassifier(categorical_features=categorical_cols_idx))])

# Data partitioning
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2,
                                                    random_state=0)
# Fit the model
hgbc_clf.fit(X_train, y_train)
print("model score: %.3f" % hgbc.score(X_test, y_test))

# Make predictions on test and create submission file

hgbc_test = hgbc_clf.predict(test_X)
submit_dict = {'id': test_X['id'],
               'status_group': hgbc_test}

hgbc_1_submission = pd.DataFrame(submit_dict, columns=['id', 'status_group'])
hgbc_1_submission.to_csv('output/hgbc_1_submission.csv', index=False)
