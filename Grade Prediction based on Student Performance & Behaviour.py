from google.colab import drive
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Students Performance Dataset.csv')

df['Grade'].value_counts()

df = df.drop(['Student_ID', 'First_Name', 'Last_Name', 'Email'], axis=1, inplace=False)

# there will be columns that would be one hot encoded because of its non-ordinal nature and there will be columns that label encodied because of its ordinal nature.

df.isnull().sum()

# because parent_education has 1025 row with null value, we remove the row that has null value for parent_education_level

df.shape

df = df.dropna(subset=["Parent_Education_Level"])

df.shape

# in order to see if the rows with null value on the parent education level subset has already been removed or not

x = df.drop('Grade', axis=1, inplace=False)
y = df['Grade']

onehot = ['Department']

label_encoding = [
    'Gender', 'Extracurricular_Activities', 'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level'
]

numerical = [
    'Age', 'Attendance (%)', 'Midterm_Score', 'Final_Score', 'Assignments_Avg',
    'Quizzes_Avg', 'Participation_Score', 'Projects_Score', 'Total_Score',
    'Study_Hours_per_Week', 'Stress_Level (1-10)', 'Sleep_Hours_per_Night'
]

# data preprocessing

ohe_encoder = OneHotEncoder()
ordinal_encoder = OrdinalEncoder()
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', ohe_encoder, onehot),
        ('label_encoding', ordinal_encoder, label_encoding),
        ('scaler', scaler, numerical)
    ]
)

x = preprocessor.fit_transform(df)

# y also get encoded
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# making xgb classifier model, using classifier model instead of regression model because the final result we want is a model that could predict whether it's A, B, dll. and not predict continous number.
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=10,
    random_state=42
)

# split the data size into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# training the model
model.fit(x_train, y_train)

# model made prediction based on test set
y_pred = model.predict(x_test)


# model evaluation
classification_report = classification_report(y_test, y_pred)
print(classification_report)

# confusion matrix with seaborn graph
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()