import numpy as np
from changepoint_detector_probe.page_hinkley import PageHinkleyDetector
from changepoint_detector_probe.run_probe import run_experiment

def test_page_hinkley_no_false_positive():
    detector = PageHinkleyDetector(delta_tol=0.05, lambda_threshold=0.5)
    rng = np.random.RandomState(42)
    # Small noise around 0
    series = rng.normal(0, 0.01, 100)
    
    alarms = 0
    for x in series:
        if detector.update(x):
            alarms += 1
            
    assert alarms == 0, "Disparou falso positivo em ruido estacionario"

def test_page_hinkley_detects_mean_shift():
    detector = PageHinkleyDetector(delta_tol=0.05, lambda_threshold=0.5)
    rng = np.random.RandomState(42)
    # 0 for 50 steps
    series1 = rng.normal(0, 0.01, 50)
    # Shift to 1.0 for 50 steps
    series2 = rng.normal(1.0, 0.01, 50)
    series = np.concatenate([series1, series2])
    
    alarms = []
    for i, x in enumerate(series):
        if detector.update(x):
            alarms.append(i)
            
    assert len(alarms) > 0, "Nao detectou a mudanca de media"
    assert alarms[0] >= 50, "Detectou mudanca de media muito cedo (falso positivo)"

def test_page_hinkley_reset():
    detector = PageHinkleyDetector(delta_tol=0.05, lambda_threshold=0.5)
    rng = np.random.RandomState(42)
    series1 = rng.normal(0, 0.01, 50)
    series2 = rng.normal(1.0, 0.01, 50)
    series = np.concatenate([series1, series2])
    
    triggered = False
    for i, x in enumerate(series):
        if detector.update(x):
            triggered = True
            break
            
    assert triggered
    assert detector.mean_estimate == 0.0
    assert detector.n == 0
    assert detector.m_t == 0.0
    assert detector.M_t == 0.0

def test_page_hinkley_deterministic():
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(42)
    
    series1 = rng1.normal(0, 0.01, 50).tolist() + rng1.normal(1.0, 0.01, 50).tolist()
    series2 = rng2.normal(0, 0.01, 50).tolist() + rng2.normal(1.0, 0.01, 50).tolist()
    
    det1 = PageHinkleyDetector(delta_tol=0.05, lambda_threshold=0.5)
    alarms1 = [i for i, x in enumerate(series1) if det1.update(x)]
    
    det2 = PageHinkleyDetector(delta_tol=0.05, lambda_threshold=0.5)
    alarms2 = [i for i, x in enumerate(series2) if det2.update(x)]
    
    assert alarms1 == alarms2

def test_run_probe_lightweight():
    # Only run one scenario to check if the script is put together right
    cp = run_experiment(sleep_enabled=True, gating_enabled=False, num_eps_per_region=5)
    assert isinstance(cp, int)
