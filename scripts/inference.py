"""
scripts/inference.py
Serves the KMeans model as a SageMaker real-time endpoint.

SageMaker calls:
  model_fn()   — once on container startup
  input_fn()   — parses each incoming HTTP request
  predict_fn() — runs prediction
  output_fn()  — serializes the response
"""

import pickle, json, os, numpy as np


FEATURES = [
    'total_spend', 'purchase_frequency', 'avg_basket_value',
    'unique_products', 'category_diversity', 'price_sensitivity'
]


def model_fn(model_dir):
    """Load model artifacts once at startup."""
    with open(f'{model_dir}/kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open(f'{model_dir}/metrics.json') as f:
        metrics = json.load(f)
    return {'kmeans': kmeans, 'persona_map': metrics['persona_map']}


def input_fn(request_body, content_type='application/json'):
    """
    Parse the incoming request body.
    Expected input format:
      {"features": [total_spend, purchase_frequency, avg_basket_value,
                    unique_products, category_diversity, price_sensitivity]}
    """
    if content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data['features']).reshape(1, -1)
    raise ValueError(f'Unsupported content type: {content_type}')


def predict_fn(input_data, model):
    """Run KMeans prediction and return cluster + persona."""
    kmeans      = model['kmeans']
    persona_map = model['persona_map']
    cluster_id  = int(kmeans.predict(input_data)[0])
    persona     = persona_map.get(str(cluster_id), 'Unknown')
    return {'cluster_id': cluster_id, 'persona': persona}


def output_fn(prediction, accept='application/json'):
    """Serialize prediction result to JSON."""
    return json.dumps(prediction), 'application/json'
