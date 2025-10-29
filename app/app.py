from flask import Flask, render_template, request, jsonify
import os
import json
import numpy as np

app = Flask(__name__, template_folder='templates')

# Try to load a PyTorch model if present
model_path = os.path.join('model', 'models', 'cav1_model.pth')
MODEL = None
try:
    import torch, torch.nn as nn
    ckpt = torch.load(model_path, map_location='cpu')
    input_dim = ckpt.get('input_dim', None)
    # define a small net to load state dict (architecture must match training script)
    class SimpleNet(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 2)
            )
        def forward(self,x): return self.net(x)
    if input_dim:
        MODEL = SimpleNet(input_dim)
        MODEL.load_state_dict(ckpt['state_dict'])
        MODEL.eval()
except Exception as e:
    print('Model load skipped or failed:', e)
    MODEL = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # expecting JSON with numeric features as a flat list or dict
    if not data:
        return jsonify({'error':'no data provided'}), 400
    # If model loaded, run prediction, else return demo random result
    if MODEL is not None:
        try:
            import torch
            if isinstance(data, dict):
                vals = np.array(list(data.values()), dtype=float).reshape(1,-1)
            else:
                vals = np.array(data, dtype=float).reshape(1,-1)
            x = torch.tensor(vals, dtype=torch.float32)
            logits = MODEL(x)
            probs = torch.softmax(logits, dim=1).detach().numpy()[0,1]
            cav_state = 'promoter' if probs>0.5 else 'suppressor'
            return jsonify({'prob_promoter': float(probs), 'cav1_state': cav_state})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        import random
        p = random.random()
        return jsonify({'prob_promoter': p, 'cav1_state': 'promoter' if p>0.5 else 'suppressor', 'note':'Demo mode — no trained model in repo.'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
