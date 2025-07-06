# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load your CSV file
df = pd.read_csv("IRIS.csv")

# Encode the target variable (species)
le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

# Separate features and labels
X = df.drop("species", axis=1)
y = df["species"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_log_pred = log_model.predict(X_test_scaled)

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, y_log_pred))
print(classification_report(y_test, y_log_pred, target_names=le.classes_))

# Decision Tree Model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_scaled, y_train)
y_tree_pred = tree_model.predict(X_test_scaled)

print("\n=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, y_tree_pred))
print(classification_report(y_test, y_tree_pred, target_names=le.classes_))

# Confusion Matrix for Decision Tree
conf_mat = confusion_matrix(y_test, y_tree_pred)
sns.heatmap(conf_mat, annot=True, cmap="YlGnBu", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree")
plt.show()
