"""Independent, documented reconstruction; NOT the original ACDSA experiment."""
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description='Reconstruct the ACDSA 2027 computational case study from a locally obtained source dataset.')
parser.add_argument('--data',type=Path,default=ROOT/'ventilator_data.xlsx',help='Local .xlsx or .csv file obtained from the original dataset source; not distributed here.')
args=parser.parse_args()
source=args.data.expanduser().resolve()
if not source.is_file():
 parser.error(f'Data file not found: {source}. Download the original data from https://github.com/Swarnodip-Nag/PMEMS and see README.md.')
if source.suffix.lower()=='.xlsx':
 try: D=pd.read_excel(source,sheet_name='Ventilators')
 except ValueError: D=pd.read_excel(source)
elif source.suffix.lower()=='.csv':
 D=pd.read_csv(source)
else:
 parser.error('Expected .xlsx or .csv input')
if 'equipment_type' not in D.columns:
 parser.error('Dataset must contain an equipment_type column')
D=D.loc[D['equipment_type'].astype(str).str.strip().str.casefold()=='ventilator'].copy()
if len(D)!=220:
 parser.error(f'Expected 220 ventilator observations for the published reconstruction, found {len(D)}. Verify dataset version and filtering.')
required={'failure_probability','cost_implications','temperature_avg','humidity_avg','last_maintenance_days','uptime_hours','sensor_1','sensor_2','vibration','voltage_fluctuation','pressure'}
missing=required-set(D.columns)
if missing: parser.error(f'Missing required columns: {sorted(missing)}')
# Preserve source row order: train_test_split depends on this ordering.

features={'Environment':['temperature_avg','humidity_avg'], 'History':['last_maintenance_days','uptime_hours'], 'Technical':['sensor_1','sensor_2','vibration','voltage_fluctuation','pressure']}
cols=sum(features.values(),[])
X=D[cols].astype(float); y=D['failure_probability'].astype(float)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model=GradientBoostingRegressor(random_state=42)
model.fit(Xtr,ytr)
pred=np.clip(model.predict(Xte),0,1)
imp={k:float(100*model.feature_importances_[[cols.index(c) for c in v]].sum()) for k,v in features.items()}
metrics={'n':len(D),'train_n':len(Xtr),'test_n':len(Xte),'mae':float(mean_absolute_error(yte,pred)),'rmse':float(np.sqrt(mean_squared_error(yte,pred))),'r2':float(r2_score(yte,pred)),'importance_percent':imp,'features':cols,'model':'sklearn GradientBoostingRegressor(default parameters except random_state=42)','split':'80/20, random_state=42'}
# Evaluate policy on held-out cases, avoiding in-sample predictions. Costs are illustrative units.
cf=D.loc[Xte.index,'cost_implications'].astype(float).to_numpy()
thresholds=np.round(np.arange(.05,.951,.05),2)
def curve(risks,preventive_cost,fail_cost):
 return np.array([np.where(risks>t,preventive_cost,risks*fail_cost).sum() for t in thresholds])
base=curve(pred,2000,cf)
metrics['baseline_optimal_threshold']=float(thresholds[base.argmin()]);metrics['baseline_minimum_cost']=float(base.min())
metrics['cost_sensitivity']={str(cp):float(thresholds[curve(pred,cp,cf).argmin()]) for cp in [1000,2000,4000,6000]}
# These scenario multipliers are NEW illustrative assumptions, not sourced from original paper.
hospital=curve(pred,2000,cf)
home=curve(pred,4000,cf*1.5)
metrics['hospital_optimal_threshold']=float(thresholds[hospital.argmin()]);metrics['home_optimal_threshold']=float(thresholds[home.argmin()])
# Environmental counterfactuals: modify environmental inputs, hold all other features constant.
scenarios={}
for label,dt,dh in [('Cool',-5,-10),('Base',0,0),('Hot',5,10)]:
 x=Xte.copy();x['temperature_avg']+=dt;x['humidity_avg']+=dh
 r=np.clip(model.predict(x),0,1)
 scenarios[label]=curve(r,4000,cf*1.5)
metrics['scenario_optimal_threshold']={k:float(thresholds[v.argmin()]) for k,v in scenarios.items()}
(ROOT/'ac_dsa_reconstruction_metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False))
plots=[('fig2_reconstruction.png',list(imp.keys()),list(imp.values()),'Feature group','Importance (%)'),('fig3_reconstruction.png',thresholds,base,'Risk threshold','Expected cost (illustrative units)'),('fig4_reconstruction.png',[1000,2000,4000,6000],list(metrics['cost_sensitivity'].values()),'Preventive cost','Optimal threshold')]
for name,x,y,xlabel,ylabel in plots:
 fig,ax=plt.subplots(figsize=(6.2,3.8)); ax.bar(x,y) if name.startswith('fig2') else ax.plot(x,y,'o-');ax.set(xlabel=xlabel,ylabel=ylabel);fig.tight_layout();fig.savefig(ROOT/name,dpi=180);plt.close(fig)
for name,curves in [('fig5_reconstruction.png',{'Hospital (assumed)':hospital,'Home (assumed)':home}),('fig6_reconstruction.png',scenarios)]:
 fig,ax=plt.subplots(figsize=(6.2,3.8))
 for label,values in curves.items():ax.plot(thresholds,values,'o-',label=label)
 ax.set(xlabel='Risk threshold',ylabel='Expected cost (illustrative units)');ax.legend();fig.tight_layout();fig.savefig(ROOT/name,dpi=180);plt.close(fig)
print(json.dumps(metrics,indent=2))
