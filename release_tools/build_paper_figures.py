"""Emit an apply_patch patch for self-contained TikZ paper figures.

Reads only published evaluation arrays. No inference or external image files.
Plain TikZ avoids a separate pgfplots dependency in local PDF exporters.
"""
from pathlib import Path
import difflib
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def axes(xlim,ylim,xticks,yticks,width,height,title,xlabel,ylabel):
    def xy(x,y):return ((x-xlim[0])/(xlim[1]-xlim[0])*width,(y-ylim[0])/(ylim[1]-ylim[0])*height)
    lines=[r'\begin{tikzpicture}[x=1cm,y=1cm,font=\scriptsize]',
        rf'\node[font=\small\bfseries,align=center] at ({width/2:.3f},{height+.4:.3f}) {{{title}}};']
    for x in xticks:
        u,_=xy(x,ylim[0]);lines.extend([rf'\draw[black!10] ({u:.3f},0)--({u:.3f},{height});',rf'\node[below=3pt] at ({u:.3f},0) {{{x:g}}};'])
    for y in yticks:
        _,v=xy(xlim[0],y);lines.extend([rf'\draw[black!10] (0,{v:.3f})--({width},{v:.3f});',rf'\node[left=3pt] at (0,{v:.3f}) {{{y:g}}};'])
    lines.extend([rf'\draw[black!55] (0,{height})--(0,0)--({width},0);',
        rf'\node[font=\small] at ({width/2:.3f},-.72) {{{xlabel}}};',
        rf'\node[font=\small,rotate=90] at (-1.05,{height/2:.3f}) {{{ylabel}}};'])
    return lines,xy

def polyline(points,xy,style):
    return r'\draw['+style+'] '+'--'.join(f'({xy(x,y)[0]:.4f},{xy(x,y)[1]:.4f})' for x,y in points)+';'

def legend(lines,width,items):
    for i,(color,style,label) in enumerate(items):
        x=.25+i*width/len(items)
        lines.extend([rf'\draw[{color},{style},thick] ({x:.3f},-1.23)--({x+.55:.3f},-1.23);',
            rf'\node[anchor=west] at ({x+.65:.3f},-1.23) {{{label}}};'])

def curves(x,y1,y2,xlim,ylim,xticks,yticks,width,height,title,xlabel,ylabel,labels,colors=('ink','routepink'),markers=False):
    lines,xy=axes(xlim,ylim,xticks,yticks,width,height,title,xlabel,ylabel)
    lines.extend([r'\begin{scope}',rf'\clip (0,0) rectangle ({width},{height});'])
    for xx,yy,color,style in [(x[0],y1,colors[0],'dashed'),(x[1],y2,colors[1],'solid')]:
        points=list(zip(xx,yy));lines.append(polyline(points,xy,color+','+style+',thick'))
        if markers:
            for a,b in points:
                u,v=xy(a,b);lines.append(rf'\fill[{color}] ({u:.4f},{v:.4f}) circle (1pt);')
    lines.append(r'\end{scope}')
    legend(lines,width,list(zip(colors,['dashed','solid'],labels)))
    lines.append(r'\end{tikzpicture}')
    return '\n'.join(lines)

def generate():
    m=json.loads((ROOT/'evaluation/daily_storm_final.json').read_text())['aggregate_equal_storm']
    with np.load(ROOT/'evaluation/release_data/fung_wong_video.npz') as z:a={k:z[k] for k in z.files}
    lead=np.arange(6,121,6)
    bars=[r'\begin{tikzpicture}[x=1cm,y=1cm,font=\scriptsize]']
    for offset,title,values,top,ticks,unit in [(0,'Mean position error',[round(m[k]['mean_track_error_km'],1) for k in ['1.1','1.2']],1000,[0,500,1000],'km; lower is better'),(7.2,'Six-hour heading error',[round(m[k]['direction_error_deg'],2) for k in ['1.1','1.2']],70,[0,20,40,60],'degrees; lower is better')]:
        bars.append(rf'\begin{{scope}}[shift={{({offset},0)}}]')
        bars.append(rf'\node[font=\small\bfseries] at (2.5,4.05) {{{title}}};')
        bars.append(rf'\node[rotate=90] at (-.75,1.75) {{{unit}}};')
        for tick in ticks:
            h=tick/top*3.5;bars.extend([rf'\draw[black!10] (0,{h:.4f})--(5,{h:.4f});',rf'\node[left] at (0,{h:.4f}) {{{tick}}};'])
        for x,value,color,label in [(1.4,values[0],'baselinegray','1.1'),(3.6,values[1],'modelteal','1.2 mean of 50')]:
            h=value/top*3.5;bars.extend([rf'\fill[{color}] ({x-.38},{0}) rectangle ({x+.38},{h:.4f});',rf'\node[above] at ({x},{h:.4f}) {{{value:g}}};',rf'\node[below=3pt] at ({x},0) {{{label}}};'])
        bars.extend([r'\draw[black!55] (0,3.5)--(0,0)--(5,0);',r'\end{scope}'])
    bars.append(r'\end{tikzpicture}')
    benchmark='\n'.join(bars)+r'\par\vspace{7mm}'+'\n'+curves([lead,lead],m['1.1']['track_error_by_lead_km'],m['1.2']['track_error_by_lead_km'],(6,120),(0,2000),[6,24,48,72,96,120],[0,500,1000,1500,2000],12.2,3.7,'Error growth across forecast lead','Forecast lead (hours)','Mean position error (km)',['Trackformer 1.1','Trackformer 1.2: mean of 50'],('baselinegray','modelteal'))
    f,o=a['forecast_lat_lon'],a['observed_lat_lon']
    case=curves([o[:,1],f[:,1]],o[:,0],f[:,0],(116,140),(9,23),[116,120,124,128,132,136,140],[10,14,18,22],12.2,7.4,'Fung-wong: geographic routes, +0 to +120 h',r'Longitude ($^\circ$E)',r'Latitude ($^\circ$N)',['Observed best track','1.2 mean of 50'],markers=True)
    case+=r'\par\vspace{5mm}'+'\n'+curves([lead,lead],a['observed_pressure_hpa'],a['central_pressure_hpa'],(6,120),(925,1005),[6,24,48,72,96,120],[940,960,980,1000],12.2,3.8,'Central pressure: intensification and weakening','Forecast lead (hours)','Central pressure (hPa)',['JMA best track (IBTrACS)','1.2 member-mean pressure'])
    lat,lon,field=a['latitude'],a['longitude'],a['regional_pressure_hpa'][7]
    ys=(lat>=10)&(lat<=22);xs=(lon>=123)&(lon<=140)
    fig,ax=plt.subplots();cs=ax.contour(lon[xs],lat[ys],field[np.ix_(ys,xs)],levels=np.arange(944,1017,4));labels=ax.clabel(cs,levels=np.arange(944,1017,8),fmt='%d',fontsize=8)
    lines,xy=axes((123,140),(10,22),[124,128,132,136,140],[10,14,18,22],11.5,8.5,'Fung-wong: model-mean MSLP at +48 h',r'Longitude ($^\circ$E)',r'Latitude ($^\circ$N)')
    lines.extend([r'\begin{scope}',r'\clip (0,0) rectangle (11.5,8.5);'])
    for level,segs in zip(cs.levels,cs.allsegs):
        for seg in segs:
            if len(seg)>=2:lines.append(polyline(seg,xy,'black!45,thin'))
    for ring in json.loads((ROOT/'evaluation/release_data/fung_wong_coastlines.json').read_text()):
        p=np.array(ring)
        if np.any((p[:,0]>=123)&(p[:,0]<=140)&(p[:,1]>=10)&(p[:,1]<=22)):lines.append(polyline(p,xy,'black!70,thin'))
    for t in labels:
        u,v=xy(*t.get_position());lines.append(rf'\node[font=\tiny,fill=white,inner sep=.5pt,text=black!70] at ({u:.4f},{v:.4f}) {{{t.get_text()}}};')
    for points,color,style in [(o[:9],'ink','dashed'),(f[:9],'routepink','solid')]:
        lines.append(polyline(points[:,::-1],xy,color+','+style+',thick'))
        u,v=xy(points[-1,1],points[-1,0]);lines.append(rf'\fill[{color}] ({u:.4f},{v:.4f}) circle (2pt);')
    lines.append(r'\end{scope}');legend(lines,11.5,[('ink','dashed','Observed route to +48 h'),('routepink','solid','Forecast route to +48 h')]);lines.append(r'\end{tikzpicture}');plt.close(fig)
    return {'fig:benchmark':benchmark,'fig:fungwong':case,'fig:pressuremap':'\n'.join(lines)}

def main():
    path=ROOT/'paper/trackformer.tex';old=path.read_text();new=old.replace('tikz,pgfplots,caption','tikz,caption')
    new=new.replace('\\pgfplotsset{compat=1.18}\n','')
    new=re.sub(r'\\pgfplotsset\{paperaxis/\.style=.*?scaled ticks=false\}\}\n','',new,flags=re.S)
    figures=generate()
    def replace(match):
        text=match.group()
        for label,picture in figures.items():
            if '\\label{'+label+'}' in text:
                if label == 'fig:pressuremap':
                    picture = picture.replace(r'\begin{tikzpicture}[x=1cm', r'\begin{tikzpicture}[scale=0.9,x=1cm')
                opening = text[:text.index('\\begin{tikzpicture}')]
                return opening+picture+'\n'+text[text.index('\\caption'):]
        return text
    new=re.sub(r'\\begin\{figure\}.*?\\end\{figure\}',replace,new,flags=re.S)
    print('*** Begin Patch\n*** Update File: '+str(path))
    for line in list(difflib.unified_diff(old.splitlines(),new.splitlines(),n=3))[2:]:print('@@' if line.startswith('@@') else line)
    print('*** End Patch')

if __name__=='__main__':main()
