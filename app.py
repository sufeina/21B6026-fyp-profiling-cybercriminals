# working original code

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import pandas as pd
import os
import joblib
import pickle
import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from fpdf import FPDF

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    preview, cleaned_preview, feature_table = None, None, None
    model_status, confusion_matrix_path, accuracy = None, None, None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'upload_csv':
            file = request.files['csv_file']
            if file.filename.endswith('.csv'):
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(path)
                session['uploaded_csv'] = path
                df = pd.read_csv(path)
                session['columns'] = df.columns.tolist()
                session['columns'] = df.columns.tolist()
                session['dataset_rows'] = df.shape[0]
                session['dataset_cols'] = df.shape[1]
                session['dataset_preview'] = df.head().to_html(classes='table table-striped', index=False)

                preview = df.head().to_html(classes='table table-striped')
                session['upload_done'] = True
                flash("Dataset uploaded!", "success")
            else:
                flash("Upload a valid CSV.", "danger")

        elif action == 'select_target':
            target_column = request.form.get('target_column')
            session['target_column'] = target_column
            session['target_selected'] = True
            df = pd.read_csv(session['uploaded_csv'])
            df.dropna(subset=[target_column], inplace=True)
            df.to_csv(os.path.join(PROCESSED_FOLDER, 'current.csv'), index=False)
            session['columns'] = df.columns.tolist()
            session['available_cols'] = [c for c in df.columns if c != target_column]

            # Create Pie Chart of Target Column
            if target_column:
                df[target_column].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='Set2')
                plt.title('Target Column Distribution')
                plt.ylabel('')
                plt.tight_layout()
                plt.savefig(os.path.join('static', 'target_piechart.png'))
                plt.close()

        elif action == 'drop_columns':
            drop = request.form.getlist('columns_to_drop')
            df = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'current.csv'))
            df.drop(columns=drop, inplace=True)
            df.to_csv(os.path.join(PROCESSED_FOLDER, 'current.csv'), index=False)
            session['available_cols'] = [c for c in df.columns if c != session['target_column']]     
            session['dropped_columns'] = drop 
            session['columns_dropped'] = True

        elif action == 'preprocess':
            try:
                df = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'current.csv'))
                y = df[session['target_column']]
                X = df.drop(columns=[session['target_column']]).fillna('Unknown')
                cat_cols = X.select_dtypes(include='object').columns
                categorical_values = {}

                for col in cat_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    categorical_values[col] = list(le.classes_)
                    safe = re.sub(r'[^A-Za-z0-9_]+', '_', col)
                    joblib.dump(le, f'{UPLOAD_FOLDER}/encoder_{safe}.pkl')

                with open(os.path.join(UPLOAD_FOLDER, 'categorical_values.pkl'), 'wb') as f:
                    pickle.dump(categorical_values, f)

                final = pd.concat([X, y], axis=1)
                final.to_csv(os.path.join(PROCESSED_FOLDER, 'preprocessed_dataset.csv'), index=False)
                cleaned_preview = final.head().to_html(classes='table table-striped')
                session['preprocessing_done'] = True

                # Fix here:
                session['dropped_columns'] = session.get('dropped_columns', [])  # default to empty list if not dropped
                session['missing_filled'] = int(df.isnull().sum().sum())

                flash("Preprocessing complete.", "success")
            except Exception as e:
                flash(f"Preprocessing failed: {e}", "danger")


        elif action == 'feature_selection':
            df = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'preprocessed_dataset.csv'))
            X, y = df.drop(columns=[session['target_column']]), df[session['target_column']]

            # LOAD your existing top 10 features
            try:
                top_features = list(joblib.load('models/feature_selection.joblib')) # <-- use your saved model
                session['selected_features'] = list(top_features)

                top_df = pd.concat([X[top_features], y], axis=1)
                top_df.to_csv(os.path.join(UPLOAD_FOLDER, 'selected_features.csv'), index=False)

                feature_table = pd.DataFrame({'Feature': top_features, 'Importance': range(10, 0, -1)})
                feature_table = feature_table.to_html(classes='table table-bordered')

                plt.figure(figsize=(8, 6))
                plt.bar(top_features, range(10, 0, -1), color='skyblue')
                plt.title("Top 10 Feature Importances (from saved model)")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join('static', 'feature_importance.png'))
                plt.close()

                session['feature_selection_done'] = True
                flash("Feature selection loaded from saved model.", "success")
            except Exception as e:
                flash(f"Feature selection failed: {e}", "danger")

        elif action == 'train_model':
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, 'selected_features.csv'))
            X, y = df.drop(columns=[session['target_column']]), df[session['target_column']]

            try:
                model = joblib.load('models/random_forest.joblib')  # <-- use your saved RF model
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                y_pred = model.predict(X_test)

                acc = round(accuracy_score(y_test, y_pred) * 100, 2)
                report = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                session['accuracy'] = acc
                session['classification_report_text'] = report

                plt.figure(figsize=(6, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join('static', 'confusion_matrix.png'))
                plt.close()

                session['model_training_done'] = True
                flash(f"Model evaluation complete. Accuracy: {acc}%", "success")

            except Exception as e:
                flash(f"Model training failed: {e}", "danger")


        elif action == 'restart_process':
            session.clear()
            for f in ['selected_features.csv', 'feature_selection.joblib', 'random_forest.joblib', 'feature_importance.png', 'confusion_matrix.png', 'current.csv', 'preprocessed_dataset.csv', 'categorical_values.pkl']:
                for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, 'static']:
                    try: os.remove(os.path.join(folder, f))
                    except: pass
            flash("Process reset.", "info")
            return redirect(url_for('index'))

    return render_template('dashboard.html',
        preview=preview,
        cleaned_preview=cleaned_preview,
        feature_table=feature_table,
        model_status=model_status,
        confusion_matrix_path=confusion_matrix_path,
        accuracy=session.get('accuracy'),
        classification_report_text=session.get('classification_report_text'),
        columns=session.get('columns'),
        available_cols=session.get('available_cols'),
        target_column=session.get('target_column'),
        target_selected=session.get('target_selected'),
        columns_dropped=session.get('columns_dropped'),
        upload_done=session.get('upload_done'),
        preprocessing_done=session.get('preprocessing_done'),
        feature_selection_done=session.get('feature_selection_done'),
        model_training_done=session.get('model_training_done')
    )

@app.route('/dashboard')
def dashboard():
    return redirect(url_for('index'))

@app.route('/go_to_predict')
def go_to_predict():
    try:
        selected_features = list(joblib.load('models/feature_selection.joblib'))  # 🔧 Load from correct path
        categorical_values = pickle.load(open(os.path.join(UPLOAD_FOLDER, 'categorical_values.pkl'), 'rb'))
        return render_template('prediction.html', selected_features=selected_features, categorical_values=categorical_values)
    except Exception as e:
        return f"Error loading prediction form: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_features = list(joblib.load('models/feature_selection.joblib'))  # 🔧 Load from correct path
        categorical_values = pickle.load(open(os.path.join(UPLOAD_FOLDER, 'categorical_values.pkl'), 'rb'))
        model = joblib.load('models/random_forest.joblib')  # 🔧 Load from correct path

        input_data = request.form.to_dict()
        input_df = pd.DataFrame([input_data])

        session['user_input'] = input_data

        for col in selected_features:
            enc_path = os.path.join(UPLOAD_FOLDER, f'encoder_{col}.pkl')
            if os.path.exists(enc_path):
                le = joblib.load(enc_path)
                input_df[col] = le.transform(input_df[col].astype(str))

        input_df = input_df[selected_features].apply(pd.to_numeric, errors='coerce').fillna(0)
        prediction = model.predict(input_df)[0]
        session['prediction_result'] = prediction

        return render_template('prediction.html', prediction_result=prediction, selected_features=selected_features, categorical_values=categorical_values)

    except Exception as e:
        return f"Prediction error: {str(e)}"


@app.route('/visualization')
def visualization():
    return render_template('data_visualization.html')


@app.route('/summary')
def summary():
    return render_template('summary.html',
        prediction_result=session.get('prediction_result'),
        accuracy=session.get('accuracy'),
        classification_report_text=session.get('classification_report_text')
    )

@app.route('/download_summary_pdf')
def download_summary_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Cyber Criminal Profiling - Summary Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)

    # Target
    pdf.cell(0, 10, f"Selected Target Feature: {session.get('target_column')}", ln=True)
    pdf.ln(5)

    # Preprocessing
    pdf.cell(0, 10, "Preprocessing Summary:", ln=True)
    pdf.multi_cell(0, 8, "- Missing values filled as 'Unknown'\n- Label Encoding applied to categorical columns.")
    pdf.ln(5)

    # Features
    pdf.cell(0, 10, "Top 10 Selected Features:", ln=True)
    features = session.get('selected_features', [])
    if features:
        for feat in features:
            pdf.cell(0, 8, f"- {feat}", ln=True)
    else:
        pdf.cell(0, 8, "No features available.", ln=True)
    pdf.ln(5)

    # Model Performance
    pdf.cell(0, 10, "Model Performance:", ln=True)
    pdf.cell(0, 8, f"- Accuracy: {session.get('accuracy', 'N/A')}%", ln=True)
    pdf.ln(5)

    pdf.cell(0, 10, "Classification Report:", ln=True)
    classification_text = session.get('classification_report_text', 'N/A')
    for line in classification_text.splitlines():
        pdf.cell(0, 6, line, ln=True)

    pdf.ln(5)
    # Prediction
    pdf.cell(0, 10, f"Prediction Result: {session.get('prediction_result', 'N/A')}", ln=True)


    # Save
    filepath = os.path.join(UPLOAD_FOLDER, 'summary_report.pdf')
    pdf.output(filepath)

    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)