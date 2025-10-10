import pickle
from src.logger import logging
from flask import Flask,request,render_template


from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipline

application=Flask(__name__)
app=application

# Route for a home page 

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')

    # POST: collect form data and predict
    try:
        # Safely parse numeric inputs; default to 0 if empty or invalid
        def parse_float(field_name):
            val = request.form.get(field_name)
            if val is None or val == "":
                return 0.0
            # remove commas and whitespace
            try:
                return float(str(val).replace(',', '').strip())
            except Exception:
                logging.warning(f"Failed to parse numeric field {field_name} with value: {val}")
                return 0.0

        # template uses lowercase field names
        applicant_income = parse_float('applicantIncome')
        coapplicant_income = parse_float('coapplicantIncome')
        loan_amount = parse_float('loanAmount')
        loan_amount_term = parse_float('loanAmountTerm')
        credit_history = parse_float('creditHistory')

        # Map form field names (from templates) to the CustomData expected names
        data = CustomData(
            Gender=request.form.get('gender'),
            Married=request.form.get('married'),
            Dependents=request.form.get('dependents'),
            Education=request.form.get('education'),
            Self_Employed=request.form.get('selfEmployed'),
            ApplicantIncome=applicant_income,
            CoapplicantIncome=coapplicant_income,
            LoanAmount=loan_amount,
            Loan_Amount_Term=loan_amount_term,
            Credit_History=credit_history,
            Property_Area=request.form.get('propertyArea'),
        )
    except Exception as e:
        # Return to index with an error message if parsing fails
        return render_template('home.html', error=str(e))

    pred_df = data.get_data_as_dataframe()
    logging.info(f"value:{pred_df}")

    predict_pipeline = PredictPipline()
    results = predict_pipeline.predict(pred_df)
    logging.info(f"value:{results}")

    # Render the index template and show the prediction result
    return render_template('home.html', 
                           result=results[0],
                           )

    

if __name__ == "__main__":
    # Run locally on port 5000
    app.run(host="127.0.0.1", port=5000)