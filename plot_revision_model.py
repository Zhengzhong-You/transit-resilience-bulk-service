"""Vector schematic of the marked route and compensating service clock."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
f,axes=plt.subplots(2,1,figsize=(7.1,3.6),gridspec_kw={'height_ratios':[1.2,1]})
a=axes[0];a.set(xlim=(0,10),ylim=(-.8,2.9));a.axis('off')
for x,label in [(1,'Origin'),(3,'Station 1'),(6,'Station $n$'),(9,'Terminal $N$')]:
 a.add_patch(Rectangle((x-.65,.35),1.3,.65,facecolor='#e9edf4',edgecolor='#384d6b',lw=.8));a.text(x,.675,label,ha='center',va='center')
for x,y in [(1.7,2.3),(3.7,5.3),(6.7,8.3)]:a.annotate('',xy=(y,.675),xytext=(x,.675),arrowprops={'arrowstyle':'->','color':'#384d6b','lw':1.2})
a.text(4.5,1.13,'travel delay $J_{n,l}$',ha='center',fontsize=9)
a.text(4.5,.58,'FIFO holds',ha='center',va='top',fontsize=9)
a.text(1,-.15,'periodic dispatch $lh$',ha='center',fontsize=9)
a.annotate(r'local arrivals $\lambda_n$',xy=(6,1.03),xytext=(6,1.7),ha='center',arrowprops={'arrowstyle':'->','lw':.9})
a.text(6,-.25,'alight $d=n$; board FCFS\navailable seats $S_{n,l}=C-G_{n,l}$',ha='center',va='center',fontsize=9)
a.annotate('',xy=(8.7,1.38),xytext=(3.3,1.38),arrowprops={'arrowstyle':'->','lw':1,'connectionstyle':'arc3,rad=-.22','color':'#a45b1b'})
a.text(5.7,2.65,'same vehicle; passenger destination $d$',ha='center',fontsize=9,color='#804714')
a.text(.1,2.7,'(a)',fontweight='bold')
a=axes[1];a.set(xlim=(0,10),ylim=(-.3,1.7));a.axis('off')
a.annotate('',xy=(9.6,.0),xytext=(.5,.0),arrowprops={'arrowstyle':'->','lw':.9});a.text(9.7,-.02,'time',va='center',fontsize=9)
planned=[1.6,4.5,7.4];actual=[2.0,6.0,7.8]
for x,y,l in zip(planned,actual,['$l-1$','$l$','$l+1$']):
 a.plot([x,x],[0,.5],color='.65',ls='--',lw=1);a.plot([y,y],[0,.85],color='#0055b3',lw=1.7);a.text(y,1.02,l,ha='center',fontsize=9)
 a.annotate('',xy=(y,.42),xytext=(x,.42),arrowprops={'arrowstyle':'->','color':'#a45b1b','lw':1})
for left,right,label in [(2,6,'long interval'),(6,7.8,'short interval')]:
 a.annotate('',xy=(right,1.38),xytext=(left,1.38),arrowprops={'arrowstyle':'<->','lw':.8});a.text((left+right)/2,1.48,label,ha='center',fontsize=9)
a.text(.1,1.53,'(b)',fontweight='bold');a.text(.55,-.25,'dashed: timetable; blue: actual service; orange: displacement',fontsize=9)
f.subplots_adjust(hspace=.3,left=.03,right=.98,top=.96,bottom=.1)
out=Path('Figures/rr2_model.pdf');out.parent.mkdir(parents=True,exist_ok=True)
f.savefig(out,bbox_inches='tight',pad_inches=.04)
