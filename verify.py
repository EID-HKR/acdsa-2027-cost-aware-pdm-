"""Independent verification checks for the reconstructed ACDSA 2027 study."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
R=Path(__file__).resolve().parent
D=pd.read_excel(R/'ventilator_data.xlsx',sheet_name='Ventilators')
assert D.shape[0]==220 and not D.isna().any().any()
features=['temperature_avg','humidity_avg','last_maintenance_days','uptime_hours','sensor_1','sensor_2','vibration','voltage_fluctuation','pressure']
X=D[features].astype(float); y=D.failure_probability.astype(float)
tr,te,yt,ye=train_test_split(X,y,test_size=.2,random_state=42)
m=GradientBoostingRegressor(random_state=42).fit(tr,yt)
p=np.clip(m.predict(te),0,1)
cf=D.loc[te.index,'cost_implications'].to_numpy(dtype=float)
ts=np.round(np.arange(.05,.951,.05),2)
def cost(t,cp,fc=cf,r=p):return float(np.where(r>t,cp,r*fc).sum())
def optimum(cp,fc=cf,r=p):return float(min(ts,key=lambda t:cost(t,cp,fc,r)))
metrics=json.loads((R/'ac_dsa_reconstruction_metrics.json').read_text())
assert abs(mean_absolute_error(ye,p)-metrics['mae'])<1e-12
assert abs(r2_score(ye,p)-metrics['r2'])<1e-12
assert abs(optimum(2000)-metrics['baseline_optimal_threshold'])<1e-12
assert abs(cost(.30,2000)-metrics['baseline_minimum_cost'])<1e-8
assert optimum(4000,cf*1.5)==metrics['home_optimal_threshold']
for cp,t in metrics['cost_sensitivity'].items():assert optimum(int(cp))==t
for label,dt,dh in [('Cool',-5,-10),('Base',0,0),('Hot',5,10)]:
 x=te.copy();x['temperature_avg']+=dt;x['humidity_avg']+=dh
 assert optimum(4000,cf*1.5,np.clip(m.predict(x),0,1))==metrics['scenario_optimal_threshold'][label]
# Algebraic equivalence of thresholded sum and device-wise action selection.
for t in ts:
 direct=sum(2000 if risk>t else risk*f for risk,f in zip(p,cf))
 assert abs(direct-cost(t,2000))<1e-8
# Unconstrained device-specific policy cannot exceed any fixed-threshold policy.
individual=float(np.minimum(2000,p*cf).sum())
assert individual<=min(cost(t,2000) for t in ts)+1e-8
out={'status':'PASS','rows':len(D),'columns':len(D.columns),'missing_values':int(D.isna().sum().sum()),'tests':['dataset shape/completeness','prediction metrics','baseline threshold and cost','sensitivity thresholds','hospital/home scenario','environment counterfactuals','independent cost summation','device-specific policy lower bound'],'device_specific_cost':individual,'common_threshold_min_cost':min(cost(t,2000) for t in ts),'note':'Internal computational verification only; synthetic labels and costs are not externally validated.'}
(R/'verification_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
