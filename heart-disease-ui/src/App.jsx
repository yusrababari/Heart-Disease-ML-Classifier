import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';
const FEATURE_LABELS = {
  age: 'Age (years)',
  sex: 'Sex (1 = Male, 0 = Female)',
  trestbps: 'Resting Blood Pressure (mm Hg)',
  chol: 'Serum Cholesterol (mg/dl)',
  fbs: 'Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)',
  thalach: 'Maximum Heart Rate Achieved',
  exang: 'Exercise Induced Angina (1 = Yes, 0 = No)',
  oldpeak: 'ST Depression (Oldpeak)',
  ca: 'Number of Major Vessels (0-3)',
  AgeGroup: 'Age Group Category',
  cp_1: 'Chest Pain: Atypical Angina',
  cp_2: 'Chest Pain: Non-anginal Pain',
  cp_3: 'Chest Pain: Asymptomatic',
  restecg_1: 'Resting ECG: ST-T Wave Abnormality',
  restecg_2: 'Resting ECG: Left Ventricular Hypertrophy',
  slope_1: 'ST Slope: Flat',
  slope_2: 'ST Slope: Downsloping',
  'thal_2.0': 'Thalassemia: Fixed Defect',
  'thal_3.0': 'Thalassemia: Reversible Defect'
};

function App() {
  const [features, setFeatures] = useState([]);
  const [formData, setFormData] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch feature list on load
  useEffect(() => {
    axios.get(`${API_URL}/features`)
      .then((res) => {
        const fetchedFeatures = res.data.features || [];
        setFeatures(fetchedFeatures);
        
        // Build initial object with strictly empty strings
        const initialForm = {};
        fetchedFeatures.forEach((feat) => {
          initialForm[feat] = '';
        });
        setFormData(initialForm);
      })
      .catch((err) => console.error('Error fetching features:', err));
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    
    // Store as string directly while typing so React doesn't convert empty values to 0
    setFormData((prevData) => ({
      ...prevData,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    // Convert string inputs to floats right before sending to backend
    const sanitizedData = {};
    Object.keys(formData).forEach((key) => {
      const parsed = parseFloat(formData[key]);
      sanitizedData[key] = isNaN(parsed) ? 0 : parsed;
    });

    try {
      const response = await axios.post(`${API_URL}/predict`, { features: sanitizedData });
      setResult(response.data);
    } catch (err) {
      console.error('Prediction failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem', fontFamily: 'sans-serif', maxWidth: '600px', margin: 'auto' }}>
      <h2>Heart Disease Prediction UI</h2>

      <form onSubmit={handleSubmit}>
        {features.map((feature) => (
          <div key={feature} style={{ marginBottom: '1rem' }}>
            <label style={{ display: 'block', fontWeight: 'bold' }}>
              {FEATURE_LABELS[feature] || feature}:
            </label>
            <input
              type="number"
              step="any"
              name={feature}
              placeholder="Enter a value..."
              value={formData[feature] !== undefined ? formData[feature] : ''}
              onChange={handleChange}
              className="custom-input"
              style={{ width: '100%', padding: '8px', marginTop: '4px' }}
            />
          </div>
        ))}
        <button type="submit" style={{ padding: '10px 20px', cursor: 'pointer' }}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {result && (
        <div style={{ marginTop: '2rem', padding: '1rem', border: '1px solid #ccc', borderRadius: '8px' }}>
          <h3>Results:</h3>
          <p><strong>Prediction:</strong> {result.prediction === 1 ? 'High Risk' : 'Low Risk'}</p>
          {result.probability !== null && (
            <p><strong>Confidence/Probability:</strong> {(result.probability * 100).toFixed(2)}%</p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;