"""
Fig. A7: principal-component-regression (PCR) diagnostics on the modeling dataset (N = 14,421).
(a) variance explained, (b) PC1-PC2 loadings (standardized PCA of the eight predictors on the full record),
(c, d) pooled out-of-fold observed vs. predicted H and LE by season (results/pcr_cv/predictions.csv),
(e) season-resolved out-of-fold R2 (results/pcr_cv/metrics.json).
"""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
from matplotlib import gridspec
import style as S
SEAS_COL=S.SEASON_COLORS
FEATS=["Rn","Ts","Ta","Precip","RH","Ws","SWC","albedo"]
S.apply()   # house style shared by all figures

def _grid(ax):
    S.grid(ax)
def _lt(ax,t): ax.set_title(t,loc="left",fontweight="bold",fontsize=10)

def build():
    m=pd.read_csv(S.paths.MODELING).rename(columns={"Fn":"Rn","WindSpeed":"Ws_"})
    m["Ws"]=m["Ws_"]; m["albedo"]=(m["Fsu"]/m["Fsd"]).clip(0,1)
    X=m[FEATS].values; Xs=(X-X.mean(0))/X.std(0,ddof=0)
    _,sv,Vt=np.linalg.svd(Xs,full_matrices=False); evr=sv**2/np.sum(sv**2); load=Vt.T[:,:2].copy()
    P=pd.read_csv(os.path.join(S.paths.RESULTS,"pcr_cv","predictions.csv"))
    M=json.load(open(os.path.join(S.paths.RESULTS,"pcr_cv","metrics.json")))
    Sm={"H":M["H"],"LE":M["LE"],"perH":M["H_season"],"perLE":M["LE_season"]}
    d={"obsH":P["H_obs"].values,"predH":P["H_mod"].values,"seasonH":P["Season"].values,
       "obsLE":P["LE_obs"].values,"predLE":P["LE_mod"].values,"seasonLE":P["Season"].values}
    scoresH=None
    if load[FEATS.index("Precip"),1]<0:   # orient PC2 so precipitation points up
        load[:,1]*=-1
    obsH,predH,seaH=d["obsH"],d["predH"],d["seasonH"]; obsLE,predLE,seaLE=d["obsLE"],d["predLE"],d["seasonLE"]
    fig=plt.figure(figsize=(S.DOUBLE_COL,8.6))
    gs=gridspec.GridSpec(3,2,height_ratios=[1.0,1.0,0.75],hspace=0.42,wspace=0.62,
                         left=0.09,right=0.93,top=0.96,bottom=0.06)
    # (a) scree
    ax=fig.add_subplot(gs[0,0]); _grid(ax); pcs=np.arange(1,len(evr)+1)
    ax.bar(pcs,evr*100,color="#4292C6",edgecolor="#333333",linewidth=0.4,zorder=3)
    cum=np.cumsum(evr)*100; ax2=ax.twinx()
    ax2.plot(pcs,cum,"-o",color="#D62728",markersize=3,linewidth=1.2,zorder=4)
    ax2.set_ylabel("Cumulative variance (%)",color="#D62728"); ax2.tick_params(axis="y",colors="#D62728")
    ax2.set_ylim(0,105); ax.tick_params(axis="y",right=False); ax2.tick_params(axis="y",left=False,right=True)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)")
    ax.set_xticks(pcs); ax.set_ylim(0,max(45,evr[0]*100+5))
    ax2.annotate(f"PC1+PC2\n= {cum[1]:.0f} %",xy=(2,cum[1]),xytext=(3.2,58),fontsize=7.5,color="#D62728",
                 arrowprops=dict(arrowstyle="->",color="#D62728",lw=1))
    _lt(ax,"(a) Variance explained")
    # (b) biplot — labels are decluttered (2-D repulsion + weak spring to the arrow tip)
    ax=fig.add_subplot(gs[0,1]); _grid(ax); th=np.linspace(0,2*np.pi,200)
    ax.axhline(0,color="0.85",lw=0.8); ax.axvline(0,color="0.85",lw=0.8)
    contrib=np.sqrt(load[:,0]**2+load[:,1]**2); cmap=mpl.colors.LinearSegmentedColormap.from_list("viridis_dark",plt.cm.viridis(np.linspace(0,0.85,256)))   # no pale yellow on white
    norm=mpl.colors.Normalize(vmin=contrib.min(),vmax=contrib.max())
    tips=load                                           # loadings plotted at their true length
    labpos=tips*1.18                                    # declutter labels
    for _ in range(400):
        disp=np.zeros_like(labpos)
        for i in range(len(labpos)):
            for j in range(i+1,len(labpos)):
                dd=labpos[i]-labpos[j]; dist=float(np.hypot(dd[0],dd[1]))
                if 1e-9<dist<0.22:
                    push=(0.22-dist)*0.5*dd/dist; disp[i]+=push; disp[j]-=push
        disp += (tips*1.18 - labpos)*0.04
        labpos += disp*0.6
    for i,f in enumerate(FEATS):
        c=cmap(norm(contrib[i])); xa,ya=tips[i]
        ax.arrow(0,0,xa,ya,head_width=0.028,head_length=0.04,fc=c,ec=c,lw=1.2,length_includes_head=True,zorder=5)
        lx,ly=labpos[i]
        if np.hypot(lx-xa*1.06, ly-ya*1.06)>0.06:
            ax.plot([xa*1.02,lx],[ya*1.02,ly],color="0.6",lw=0.6,zorder=4)
        ax.text(lx,ly,{"Rn":r"$R_\mathrm{n}$","Ts":r"$T_\mathrm{soil}$","Ta":r"$T_\mathrm{a}$","Precip":"$P$","albedo":r"$\alpha$"}.get(f,f),fontsize=7.5,ha="center",va="center",fontweight="bold",color=c,zorder=6,
                bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none",alpha=0.75))
    ax.set_xlim(-1.0,1.0); ax.set_ylim(-1.0,1.0); ax.set_aspect("equal")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f} %)"); ax.set_ylabel(f"PC2 ({evr[1]*100:.1f} %)")
    smap=plt.cm.ScalarMappable(cmap=cmap,norm=norm); smap.set_array([])
    cb=fig.colorbar(smap,ax=ax,fraction=0.046,pad=0.04); cb.set_label("Loading vector length",fontsize=7.5); cb.ax.tick_params(labelsize=7)
    _lt(ax,"(b) PC1–PC2 loadings")
    # (c,d)
    def scat(ax,obs,pred,sea,stat,flux):
        _grid(ax)
        for s in ["MAM","JJA","SON","DJF"]:
            m=sea==s; ax.scatter(obs[m],pred[m],s=2,c=SEAS_COL[s],alpha=0.35,edgecolors="none",zorder=3,label=s)
        lim=[-160,700] if flux=="H" else [-120,420]
        ax.plot(lim,lim,"k-",lw=1.0,zorder=4,label="1:1"); b=np.polyfit(obs,pred,1); xs=np.array(lim)
        ax.plot(xs,b[0]*xs+b[1],"--",color="k",lw=1.2,zorder=4,label="OLS fit")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel(f"Observed ${flux}$ (W m$^{{-2}}$)"); ax.set_ylabel(f"PCR-predicted ${flux}$ (W m$^{{-2}}$)")
        bias_s=f"{stat['Bias']:+.2f}".replace("-","\u2212")
        txt=(f"$R^2$ = {stat['R2']:.2f}\nRMSE = {stat['RMSE']:.1f} W m$^{{-2}}$\nBias = {bias_s} W m$^{{-2}}$\nSlope = {stat['Slope']:.2f}\n"
             f"$N$ = {stat['N']:,}")
        ax.text(0.04,0.96,txt,transform=ax.transAxes,va="top",ha="left",fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3",fc="white",ec="#888",lw=0.6,alpha=0.95),zorder=6)
        ax.legend(loc="lower right",fontsize=7,frameon=True,framealpha=0.95,markerscale=3,handletextpad=0.3)
    axc=fig.add_subplot(gs[1,0]); scat(axc,obsH,predH,seaH,Sm["H"],"H"); _lt(axc,"(c) $H$: observed vs. predicted")
    axd=fig.add_subplot(gs[1,1]); scat(axd,obsLE,predLE,seaLE,Sm["LE"],"LE"); _lt(axd,"(d) $LE$: observed vs. predicted")
    # (e)
    axe=fig.add_subplot(gs[2,:]); _grid(axe); seasons=["MAM","JJA","SON","DJF"]
    r2H=[Sm["perH"][s]["R2"] for s in seasons]; r2LE=[Sm["perLE"][s]["R2"] for s in seasons]
    x=np.arange(len(seasons)); w=0.36
    for xs,vals,col,lab in [(x-w/2,r2H,S.FLUX_COLORS["H"],"$H$"),(x+w/2,r2LE,S.FLUX_COLORS["LE"],"$LE$")]:
        bars=axe.bar(xs,vals,w,label=lab,color=col,edgecolor="#333333",linewidth=0.4,zorder=3)
        for bar in bars: axe.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.012,f"{bar.get_height():.2f}",ha="center",va="bottom",fontsize=7)
    axe.set_xticks(x); axe.set_xticklabels(seasons); axe.set_ylabel("$R^2$ (out-of-fold)")
    axe.set_ylim(0,0.85); axe.set_xlabel("Season"); axe.legend(loc="upper right",fontsize=7.5,frameon=True,framealpha=0.95)
    _lt(axe,"(e) Season-resolved PCR skill")
    fig.savefig(S.paths.fig("FigA7_PCR_Diagnostics.png"),dpi=600,bbox_inches="tight",pad_inches=0.04); plt.close(fig)
    print("-> figures/output/FigA7_PCR_Diagnostics.png  PC1 %.1f %%, PC2 %.1f %%" % (100*evr[0], 100*evr[1]))

if __name__=="__main__":
    build()
