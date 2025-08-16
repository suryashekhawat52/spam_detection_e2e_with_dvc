from flask import Flask, request, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])

def predict_datapoints():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data = CustomData(
            message= request.form.get('message')
        )
        final_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_df)

        results = pred
        
        return render_template('results.html', final_result = results)



if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug =True)
