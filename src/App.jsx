import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL || '/api';

const FORM_FIELDS = [
  {
    section: 'Demographics',
    blurb: 'Basic patient information',
    fields: [
      {
        key: 'age', label: 'Age', type: 'number', placeholder: 'e.g. 56',
        min: 18, max: 100, hint: 'years',
      },
      {
        key: 'sex', label: 'Sex', type: 'select', default: '1',
        options: [
          { value: '1', label: 'Male' },
          { value: '0', label: 'Female' },
        ],
      },
      {
        key: 'weight', label: 'Weight', type: 'number', placeholder: 'e.g. 78',
        min: 30, max: 200, hint: 'kg',
      },
    ],
  },
  {
    section: 'Clinical Indicators',
    blurb: 'Measurements & test results',
    fields: [
      {
        key: 'trestbps', label: 'Resting Blood Pressure', type: 'number',
        placeholder: 'e.g. 130', min: 80, max: 220, hint: 'mm Hg',
      },
      {
        key: 'chol', label: 'Serum Cholesterol', type: 'number',
        placeholder: 'e.g. 240', min: 100, max: 500, hint: 'mg/dl',
      },
      {
        key: 'thalach', label: 'Max Heart Rate', type: 'number',
        placeholder: 'e.g. 150', min: 60, max: 220, hint: 'achieved',
      },
      {
        key: 'oldpeak', label: 'ST Depression (Oldpeak)', type: 'number',
        placeholder: 'e.g. 1.2', min: 0, max: 7, step: '0.1',
      },
      {
        key: 'ca', label: 'Major Vessels (0-3)', type: 'select', default: '0',
        options: [
          { value: '0', label: '0' },
          { value: '1', label: '1' },
          { value: '2', label: '2' },
          { value: '3', label: '3' },
        ],
      },
      {
        key: 'cp', label: 'Chest Pain Type', type: 'select', default: '0',
        options: [
          { value: '0', label: 'Typical Angina' },
          { value: '1', label: 'Atypical Angina' },
          { value: '2', label: 'Non-anginal Pain' },
          { value: '3', label: 'Asymptomatic' },
        ],
      },
      {
        key: 'restecg', label: 'Resting ECG', type: 'select', default: '0',
        options: [
          { value: '0', label: 'Normal' },
          { value: '1', label: 'ST-T Abnormality' },
          { value: '2', label: 'LV Hypertrophy' },
        ],
      },
      {
        key: 'slope', label: 'ST Slope', type: 'select', default: '0',
        options: [
          { value: '0', label: 'Upsloping' },
          { value: '1', label: 'Flat' },
          { value: '2', label: 'Downsloping' },
        ],
      },
      {
        key: 'thal', label: 'Thalassemia', type: 'select', default: '0',
        options: [
          { value: '0', label: 'Normal' },
          { value: '1', label: 'Fixed Defect' },
          { value: '2', label: 'Reversible Defect' },
          { value: '3', label: 'Unknown/Other' },
        ],
      },
      {
        key: 'fbs', label: 'Fasting Blood Sugar > 120', type: 'select', default: '0',
        options: [
          { value: '0', label: 'No' },
          { value: '1', label: 'Yes' },
        ],
      },
      {
        key: 'exang', label: 'Exercise Induced Angina', type: 'select', default: '0',
        options: [
          { value: '0', label: 'No' },
          { value: '1', label: 'Yes' },
        ],
      },
    ],
  },
  {
    section: 'Lifestyle & Risk Factors',
    blurb: 'New factors now considered by the model',
    fields: [
      {
        key: 'smoking', label: 'Smoking', type: 'select', default: '0',
        options: [
          { value: '0', label: 'Non-Smoker' },
          { value: '1', label: 'Smoker' },
        ],
      },
      {
        key: 'diabetes', label: 'Diabetes', type: 'select', default: '0',
        options: [
          { value: '0', label: 'No Diabetes' },
          { value: '1', label: 'Diabetic' },
        ],
      },
    ],
  },
];

function buildInitialForm() {
  const form = {};
  FORM_FIELDS.forEach(({ fields }) => {
    fields.forEach((field) => {
      form[field.key] = field.default !== undefined ? field.default : '';
    });
  });
  return form;
}

function buildPayload(form) {
  const num = (v) => {
    const n = parseFloat(v);
    return Number.isNaN(n) ? 0 : n;
  };

  const payload = {
    age: num(form.age),
    sex: num(form.sex),
    weight: num(form.weight),
    smoking: num(form.smoking),
    diabetes: num(form.diabetes),
    trestbps: num(form.trestbps),
    chol: num(form.chol),
    fbs: num(form.fbs),
    thalach: num(form.thalach),
    exang: num(form.exang),
    oldpeak: num(form.oldpeak),
    ca: num(form.ca),
    AgeGroup: num(form.age) <= 40 ? 0 : num(form.age) <= 55 ? 1 : 2,
    cp_1: num(form.cp) === 1 ? 1 : 0,
    cp_2: num(form.cp) === 2 ? 1 : 0,
    cp_3: num(form.cp) === 3 ? 1 : 0,
    restecg_1: num(form.restecg) === 1 ? 1 : 0,
    restecg_2: num(form.restecg) === 2 ? 1 : 0,
    slope_1: num(form.slope) === 1 ? 1 : 0,
    slope_2: num(form.slope) === 2 ? 1 : 0,
    'thal_2.0': num(form.thal) === 2 ? 1 : 0,
    'thal_3.0': num(form.thal) === 3 ? 1 : 0,
  };
  return payload;
}

function HeartIcon() {
  return (
    <svg viewBox="0 0 24 24" width="26" height="26" fill="none" aria-hidden="true">
      <path
        d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"
        fill="url(#heartGrad)"
      />
      <defs>
        <linearGradient id="heartGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#e879f9" />
          <stop offset="100%" stopColor="#a855f7" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function ChartsIcon() {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">
      <path d="M4 13h6v8H4v-8zM10 3h6v18h-6V3zM16 8h6v13h-6V8z" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" />
    </svg>
  );
}

function Field({ field, value, onChange }) {
  const id = `field-${field.key}`;
  const common = {
    id,
    name: field.key,
    value,
    onChange,
  };

  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>
        {field.label}
        {field.hint && <span className="field-hint">{field.hint}</span>}
      </label>
      {field.type === 'select' ? (
        <select className="field-input" {...common}>
          {field.options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      ) : (
        <input
          className="field-input"
          type="number"
          step={field.step || 'any'}
          min={field.min}
          max={field.max}
          placeholder={field.placeholder}
          {...common}
        />
      )}
    </div>
  );
}

function ResultCard({ result }) {
  const risk = result.prediction === 1;
  const probability = result.probability !== null && result.probability !== undefined
    ? result.probability * 100
    : null;

  return (
    <div className={`result-card ${risk ? 'result-high' : 'result-low'}`}>
      <div className="result-top">
        <span className={`result-pill ${risk ? 'pill-high' : 'pill-low'}`}>
          {risk ? 'High Risk' : 'Low Risk'}
        </span>
        <span className="result-title">
          {risk ? 'Heart disease likely present' : 'Heart disease unlikely'}
        </span>
      </div>
      {probability !== null && (
        <div className="gauge">
          <div className="gauge-head">
            <span className="gauge-label">Confidence / Probability</span>
            <span className="gauge-value">{probability.toFixed(1)}%</span>
          </div>
          <div className="gauge-track">
            <div
              className="gauge-fill"
              style={{ width: `${Math.max(0, Math.min(100, probability))}%` }}
            />
          </div>
        </div>
      )}
      <p className="result-note">
        Prediction generated by a LightGBM model trained on enriched clinical data
        including weight, smoking and diabetes.
      </p>
    </div>
  );
}

function GraphsModal({ open, onClose }) {
  const [graphs, setGraphs] = useState(null);
  const [error, setError] = useState(null);
  const loaded = useRef(false);

  const fetchGraphs = useCallback(() => {
    if (loaded.current) return;
    loaded.current = true;
    setError(null);
    axios
      .get(`${API_URL}/graphs`)
      .then((res) => setGraphs(res.data.graphs || []))
      .catch((err) => {
        console.error('Failed to load graphs:', err);
        setError('Could not load graphs. Make sure the backend server is running.');
      });
  }, []);

  useEffect(() => {
    if (open) fetchGraphs();
  }, [open, fetchGraphs]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose();
    };
    if (open) {
      window.addEventListener('keydown', onKey);
      document.body.style.overflow = 'hidden';
    }
    return () => {
      window.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2 className="modal-title">Notebook Visualizations</h2>
            <p className="modal-subtitle">
              All analysis charts regenerated from the training data
            </p>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close graphs">
            <CloseIcon />
          </button>
        </div>

        <div className="modal-body">
          {error ? (
            <div className="modal-state">
              <p>{error}</p>
            </div>
          ) : !graphs ? (
            <div className="modal-state">
              <div className="spinner" />
              <p>Generating charts, this may take a few seconds...</p>
            </div>
          ) : (
            <div className="graph-grid">
              {graphs.map((graph) => (
                <figure key={graph.title} className="graph-card">
                  <img src={graph.image} alt={graph.title} loading="lazy" />
                  <figcaption>{graph.title}</figcaption>
                </figure>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [formData, setFormData] = useState(buildInitialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [graphsOpen, setGraphsOpen] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/predict`, { features: buildPayload(formData) });
      setResult(response.data);
    } catch (err) {
      console.error('Prediction failed:', err);
      setError('Prediction failed. Make sure the backend server is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData(buildInitialForm());
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="brand">
            <div className="brand-icon">
              <HeartIcon />
            </div>
            <div>
              <h1 className="brand-title">HeartScope</h1>
              <p className="brand-subtitle">Intelligent heart disease risk predictor</p>
            </div>
          </div>
          <button className="graphs-button" onClick={() => setGraphsOpen(true)}>
            <ChartsIcon />
            <span>View All Graphs</span>
          </button>
        </div>
      </header>

      <main className="main">
        <div className="hero-copy">
          <h2 className="hero-title">Assess your heart health</h2>
          <p className="hero-subtitle">
            Fill in the details below and our machine learning model will estimate your risk of heart disease.
          </p>
        </div>

        <form className="form" onSubmit={handleSubmit}>
          {FORM_FIELDS.map((group) => (
            <section className="section" key={group.section}>
              <div className="section-head">
                <span className="section-marker" />
                <div>
                  <h3 className="section-title">{group.section}</h3>
                  <p className="section-blurb">{group.blurb}</p>
                </div>
              </div>
              <div className="section-grid">
                {group.fields.map((field) => (
                  <Field
                    key={field.key}
                    field={field}
                    value={formData[field.key]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            </section>
          ))}

          {error && <div className="form-error">{error}</div>}

          <div className="form-actions">
            <button type="submit" className="primary-button" disabled={loading}>
              {loading ? (
                <span className="button-loading">
                  <span className="spinner spinner-inline" />
                  Predicting...
                </span>
              ) : (
                'Predict Risk'
              )}
            </button>
            <button type="button" className="ghost-button" onClick={handleReset}>
              Reset
            </button>
          </div>
        </form>

        {result && <ResultCard result={result} />}
      </main>

      <footer className="footer">
        <p>Powered by LightGBM · FastAPI · React — visualization data regenerated live from the notebook</p>
      </footer>

      <GraphsModal open={graphsOpen} onClose={() => setGraphsOpen(false)} />
    </div>
  );
}

export default App;
