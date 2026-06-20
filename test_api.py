import warnings
warnings.filterwarnings('ignore')
from fastapi.testclient import TestClient
from api import app

with TestClient(app) as client:
    # Test 1: tree_fall (high risk cause)
    data1 = {
        'latitude': 12.9716, 'longitude': 77.5946, 'hour': 9,
        'day_of_week': 0, 'month': 6, 'event_cause': 'tree_fall',
        'event_type': 'unplanned', 'veh_type': 'others',
        'zone': 'Central Zone 2', 'corridor': 'Non-corridor',
        'police_station': 'Cubbon Park'
    }
    response1 = client.post('/predict', json=data1)
    
    # Test 2: pot_holes (low risk cause)
    data2 = dict(data1)
    data2['event_cause'] = 'pot_holes'
    response2 = client.post('/predict', json=data2)

    print('--- tree_fall ---')
    if response1.status_code == 200:
        res1 = response1.json()
        print(f"Closure Risk: {res1['closure_risk']:.4f} ({res1['closure_label']})")
        print(f"Duration Est: {res1['estimated_duration_min']} min")
    else:
        print('Error:', response1.text)

    print('\n--- pot_holes ---')
    if response2.status_code == 200:
        res2 = response2.json()
        print(f"Closure Risk: {res2['closure_risk']:.4f} ({res2['closure_label']})")
        print(f"Duration Est: {res2['estimated_duration_min']} min")
    else:
        print('Error:', response2.text)
