import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask import redirect, url_for
# Load the RandomForestClassifier model
ranforest = pickle.load(open('ranforest.pkl', 'rb'))

# Create the Flask application
app = Flask(__name__)

# Configure the database connection
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///customerchurnpred.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the CustomerData model
class CustomerData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    SeniorCitizen = db.Column(db.Float)
    Partner = db.Column(db.Float)
    Dependents = db.Column(db.Float)
    tenure = db.Column(db.Float)
    OnlineSecurity = db.Column(db.Float)
    OnlineBackup = db.Column(db.Float)
    DeviceProtection = db.Column(db.Float)
    TechSupport = db.Column(db.Float)
    Contract = db.Column(db.Float)
    PaperlessBilling = db.Column(db.Float)
    PaymentMethod = db.Column(db.Float)
    MonthlyCharges = db.Column(db.Float)
    TotalCharges = db.Column(db.Float)
    TotalAmountPaid = db.Column(db.Float)
    AdditionalServices = db.Column(db.Float)
    TenureCategory = db.Column(db.Float)
    MonthlyToTotalRatio = db.Column(db.Float)

@app.route('/', methods=['GET', 'POST'])
def churn_prediction():
    if request.method == 'POST':
        # Extract form data
        senior_citizen = int(request.form['SeniorCitizen'])
        partner = int(request.form['Partner'])
        dependents = int(request.form['Dependents'])
        tenure = int(request.form['tenure'])
        online_security = int(request.form['OnlineSecurity'])
        online_backup = int(request.form['OnlineBackup'])
        device_protection = int(request.form['DeviceProtection'])
        tech_support = int(request.form['TechSupport'])
        contract = int(request.form['Contract'])
        paperless_billing = int(request.form['PaperlessBilling'])
        payment_method = int(request.form['PaymentMethod'])
        monthly_charges = float(request.form['MonthlyCharges'])
        total_charges = float(request.form['TotalCharges'])
        total_amount_paid = float(request.form['TotalAmountPaid'])
        additional_services = int(request.form['AdditionalServices'])
        tenure_category = int(request.form['TenureCategory'])
        monthly_to_total_ratio = float(request.form['MonthlyToTotalRatio'])

        # Perform prediction
        input_data = np.array([[senior_citizen, partner, dependents, tenure, online_security, online_backup,
                                device_protection, tech_support, contract, paperless_billing, payment_method,
                                monthly_charges, total_charges, total_amount_paid, additional_services,
                                tenure_category, monthly_to_total_ratio]])

        prediction = ranforest.predict(input_data)[0]
        prediction_result = "Churn" if prediction == 1 else "Not Churn"

        # Add form data into a dictionary
        data = {
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'TotalAmountPaid': total_amount_paid,
            'AdditionalServices': additional_services,
            'TenureCategory': tenure_category,
            'MonthlyToTotalRatio': monthly_to_total_ratio,
        }

        # Insert data into SQLite database
        new_data = CustomerData(**data)
        db.session.add(new_data)
        db.session.commit()

        # Redirect to prediction result page with the prediction text
        return redirect(url_for('prediction_result', prediction_result=prediction_result))

    return render_template('home.html')

@app.route('/prediction_result/<prediction_result>', methods=['GET'])
def prediction_result(prediction_result):
    return render_template('prediction_result.html', prediction_text="Customer Churn Prediction = {}".format(prediction_result))
# Add this route to your Flask application
@app.route('/personalized_offers')
def personalized_offers():
    return render_template('personalized_offers.html')
@app.route('/feedback_surveys')
def feedback_surveys():
    return render_template('feedback_surveys.html')
@app.route('/enhanced_support')
def enhanced_support():
    return render_template('enhanced_support.html')

# API route for making predictions
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    new_data = np.array(list(data.values())).reshape(1, -1)
    output = ranforest.predict(new_data)
    if output == 0:
        result = "Not Churn"
    else:
        result = "Churn"
    return jsonify({"Prediction": result})
# Route for rendering home page with prediction result
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1, -1)
    output = ranforest.predict(final_input)[0]
    if output == 0:
        result = "Not Churn"
    else:
        result = "Churn"
    return render_template("home.html", prediction_text="The Customer Churn prediction = {}".format(result))

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
