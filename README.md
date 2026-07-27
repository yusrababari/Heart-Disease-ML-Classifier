<div align="center">


 
<table align="center" width="100%">
<tr>
<td style="background:#2b0a1f; border-radius:12px;" width="60%">
 
**Heart Disease Classifier:** FastAPI endpoint for making heart disease predictions from clinical features.
  
 - Model training is handled inside the container at build time.
 - Prediction API available at `/predict` and `/batch_predict`.
 - Build with `docker build -t heart-disease-classifier .` and run with `docker run --rm -p 80:80 heart-disease-classifier`.
  
 Example request (use the long form dataset feature names):
 ```bash
 curl -X POST http://localhost/predict \
   -H "Content-Type: application/json" \
   -d '{
     "age": 63,
     "sex": 1,
     "chest": 4,
     "resting_blood_pressure": 145,
     "serum_cholestoral": 233,
     "fasting_blood_sugar": 1,
     "resting_electrocardiographic_results": 2,
     "maximum_heart_rate_achieved": 150,
     "exercise_induced_angina": 0,
     "oldpeak": 2.3,
     "slope": 2,
     "number_of_major_vessels": 0,
     "thal": 6
   }'
 ```
 
</td>
</tr>
</table>
 
<table align="center" width="100%">
<tr>
<td style="background:#2b0a1f; border-radius:12px;" width="60%">
 

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:ec4899,100:831843&height=100&section=footer" width="100%"/>

</div>
