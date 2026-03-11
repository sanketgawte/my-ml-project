from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
# Multiple Methods
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # 1. DATA AUDIT
        original_shape = df.shape
        null_report = df.isnull().sum().to_dict()
        
        # 2. TARGET & CLEANING
        if 'views' in df.columns:
            df['target'] = np.where(df['views'] > df['views'].median(), 1, 0)
        else:
            df['target'] = LabelEncoder().fit_transform(df.iloc[:, -1].astype(str))

        cols_to_drop = [c for c in df.columns if any(word in c.lower() for word in ['id', 'title', 'tag', 'url', 'thumbnail'])]
        df = df.drop(columns=[c for c in cols_to_drop if c != 'target'])

        X = df.drop(columns=['target'])
        Y = df['target']

        # 3. PREPROCESSING
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            else:
                X[col] = X[col].fillna(X[col].mean())

        X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.2, random_state=0)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 4. MULTI-METHOD TRAINING
        models = {
            "Decision Tree": DecisionTreeClassifier(criterion='entropy', random_state=0),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=0),
            "SVM": SVC(kernel='linear', random_state=0),
            "Logistic Regression": LogisticRegression()
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, Y_train)
            acc = accuracy_score(Y_test, model.predict(X_test))
            results[name] = round(acc * 100, 2)

        return render_template('index.html', 
                               results=results, 
                               shape=original_shape,
                               nulls=null_report,
                               total_nulls=sum(null_report.values()),
                               features=X.columns.tolist())

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)