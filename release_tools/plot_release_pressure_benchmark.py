"""Plot matched central-pressure MAE and unshifted pressure-versus-time curves."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'evaluation/released_daily/released_daily_benchmark.json').read_text())
assert data['status']=='complete_verified'
fig,axes=plt.subplots(2,2,figsize=(12,8))
colors=['#66758b','#007d78']
for column,agency in enumerate(('JMA','USA')):
    metric=next(m for m in data['metrics'] if m['key']=='pressure_'+agency+'_hpa')
    assert metric['coverage']['storms']==40
    axis=axes[0,column]
    values=np.array([metric['values'][m] for m in ('1.1','1.2')])
    intervals=np.array([metric['intervals'][m] for m in ('1.1','1.2')])
    errors=np.stack([values-intervals[:,0],intervals[:,1]-values])
    bars=axis.bar(['1.1','1.2 · mean of 50'],values,yerr=errors,color=colors,width=.52,capsize=5)
    axis.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=5,fontweight='bold')
    axis.set_ylim(0,20)
    axis.set_title(f"{agency} pressure MAE · {metric['coverage']['daily_issues']} shared starts",loc='left',fontweight='bold',fontsize=12)
    axis.set_ylabel('Central-pressure MAE (hPa) · lower is better')
    lower=axes[1,column]
    issue=data['pressure_curves'][0]
    observed=np.array([np.nan if v is None else v for v in issue['observed'][agency]])
    lower.plot(data['cohort']['leads_hours'],observed,label=agency+' best track',color='#253643',linewidth=2.2)
    for model,color,style in zip(('1.1','1.2'),colors,('--','-')):
        lower.plot(data['cohort']['leads_hours'],issue['forecasts'][model],label=model,color=color,linestyle=style,linewidth=2.2)
    lower.set_xticks([6,24,48,72,96,120])
    lower.set_title(issue['name']+' · '+issue['issue_time_utc'][:16].replace('T',' ')+' UTC',loc='left',fontsize=10)
    lower.set_xlabel('Forecast lead (hours)')
    lower.set_ylabel('Actual central pressure (hPa)')
    lower.legend(frameon=False)
    for a in (axis,lower):
        a.spines[['top','right']].set_visible(False)
        a.set_axisbelow(True);a.grid(axis='y',alpha=.15)
fig.suptitle('Trackformer 1.1 vs 1.2 | matched central pressure',x=.075,ha='left',fontsize=17,fontweight='bold')
fig.text(.075,.923,'Same exact valid times · daily → storm → equal storm · 40 storms · JMA and USA separate',color='#526174')
fig.text(.075,.025,'Slight descriptive mean improvement; paired difference intervals include zero. 95% whole-storm intervals.\nOverlay: first eligible frozen case, not a best example. Exact-time physical hPa curves; no shift or warping. Missing is not zero.',fontsize=9,color='#526174')
fig.subplots_adjust(left=.075,right=.98,top=.85,bottom=.12,hspace=.43,wspace=.25)
fig.savefig(ROOT/'evaluation/released_daily/pressure_comparison.png',dpi=180,facecolor='white')
