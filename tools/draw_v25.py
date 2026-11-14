"""Generate editable draw.io and SVG previews from one explicit geometry model."""
from pathlib import Path
import html
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/architecture_v25'
PALETTE = {'blue':('#e6f0fc','#5283b5'), 'pink':('#f8e6ee','#b87394'),
           'green':('#e6f3eb','#4b9270'), 'orange':('#fff0dc','#bf8b47'),
           'gray':('#edf0f4','#778495')}
SHAPES = {'box':'rounded=1;', 'tensor':'shape=cube;size=14;',
          'input':'shape=parallelogram;', 'decision':'rhombus;',
          'document':'shape=document;', 'ellipse':'ellipse;'}


def page(name, title, subtitle, nodes, edges, notes):
    return dict(name=name, title=title, subtitle=subtitle, nodes=nodes, edges=edges, notes=notes)


# node: id, text, x,y,w,h, color, shape. edge: source,target,ports,label,waypoints.
PAGES = [
page('01-model','VOSR v2.5 | Observable Continuous Transport',
     'Frozen generation + explicit geometry solver | Implementation prototype; natural-image benefit unverified', [
('lr','LR observation\ny  /  RGB',60,220,190,110,'blue','input'),
('generator','Frozen VOSR\n25 Euler steps\nseed = 6',330,200,240,150,'pink','box'),
('native','Native RGB\nx0',660,220,200,110,'pink','tensor'),
('solve','Observation-conditioned solver\nRGB camera residual\n18 continuous coefficients',970,200,300,150,'blue','box'),
('lr_shared','Same LR\ny',1360,220,130,110,'blue','input'),
('field','Whole-image velocity field\nv = B a\nHR-pixel coordinates',970,460,300,140,'green','tensor'),
('warp','Explicit transport\nexp(-v) + RGB sampling\nNo rectangular cut-and-paste',660,460,240,140,'green','box'),
('gate','Residual descent\n+ discrete checks?',350,450,220,160,'orange','decision'),
('final','Final RGB\naccepted / retained',60,470,200,120,'green','document'),
('audit','Stage record\nnative / last revised / final\nrank, motion, rejection reason',640,740,360,100,'gray','document'),
('limits','Scope: smooth geometry only\nNo missing-content or topology repair\nNo HR or annotation at inference',1060,740,400,110,'gray','box')], [
('lr','generator','R L','',[]),('generator','native','R L','one decode',[]),
('native','solve','R L','current geometry',[]),
('lr_shared','solve','L R','',[]),
('solve','field','B T','continuous command',[]),('field','warp','L R','',[]),
('native','warp','B T','material source',[]),('warp','gate','L R','proposal',[]),
('gate','final','L R','accept',[]),
('native','final','T L','retain when rejected',[(760,170),(30,170),(30,530)]),
('gate','audit','B T','decision record',[(460,710),(820,710)])],
['Spatial evidence  /  blue','Frozen generator  /  pink','Continuous geometry  /  green','Evaluation & decisions  /  amber']),
page('02-solver','From RGB evidence to an executable coordinate map',
     'model.py + geometry.py | Fixed 3 iterations; four backtracking rates; no learned residual adapter', [
('coeff','Coefficients a\n18 floats',60,220,190,100,'blue','ellipse'),
('basis','Smooth whole-image basis\n3 x 3 cosine modes / axis\nv = B a',330,200,260,140,'green','tensor'),
('integrate','Backward flow integration\nd0 = -v / 64\nd <- d + sample(d,d), 6 times',680,200,340,140,'green','box'),
('sample','RGB resampling\nx(a) = x0(p + d)',1130,220,280,100,'green','box'),
('camera','Frozen camera A\nsame operator for all trials',1130,450,280,120,'gray','box'),
('profile','Bounded RGB gain / bias\npositive gain: 0.5 to 2.0\ncentered residual; fixed crop',680,440,340,140,'blue','box'),
('svd','Finite-difference Jacobian\nJ = U diag(s) V^T\nretain s * delta > sigma',330,440,260,140,'blue','box'),
('update','Damped step\ncap motion\nfixed line search',60,440,210,140,'orange','box'),
('safe','Safe descent?',70,710,190,120,'orange','decision'),
('keep','Accept / retain current a\nrank = 0 at start -> native',350,710,360,120,'green','box'),
('caution','Sensitivity is not probability\nJacobian > 0 is a discrete check\nLR descent is not HR truth',860,710,550,120,'gray','document')], [
('coeff','basis','R L','',[]),('basis','integrate','R L','HR units',[]),
('integrate','sample','R L','backward map',[]),('sample','camera','B T','',[]),
('camera','profile','L R','prediction + LR',[]),('profile','svd','L R','r(a)',[]),
('svd','update','L R','modes',[]),('update','safe','B T','trial',[]),
('safe','keep','R L','decision',[]),
('keep','coeff','B B','next iteration',[(530,890),(30,890),(30,365),(155,365)])],
['Finite difference = 0.125 HR px','Default resolution = 0.25 HR px','All coefficients float; zero command is exact identity']),
page('03-evaluation','Does super-resolution help geometric correspondence?',
     'Implemented planar benchmark + explicit route toward 3D | One generation seed; no manual blind scoring', [
('source','Server-side source images\nphysical parent identity',60,220,260,120,'gray','document'),
('views','Two views + known homography\nautomatically defined reference\nwarp HR before degradation',420,200,330,160,'blue','input'),
('lrpair','Same camera degradation\nLR pair',870,220,260,120,'gray','box'),
('vosr','Frozen VOSR\nseed = 6',1250,220,230,120,'pink','box'),
('controls','Paired methods\nnative | translation only\nsmooth unfiltered | full v2.5',1120,470,360,150,'green','box'),
('match','Fixed SIFT + RANSAC\nsame matcher for every method\nno descriptors / no fit = failure',620,470,380,150,'blue','box'),
('auc','Homography AUC @ 3 / 5 / 10 px\nfit failure rate; all-pair denominator\nparent-macro + pair summary',60,470,450,150,'orange','document'),
('iqa','Secondary checks\nRGB PSNR / LPIPS / DISTS\ngeometry latency + coverage',1100,750,380,110,'orange','box'),
('extend','Future external confirmation\nfixed LightGlue + calibrated views\npose AUC @ 5 / 10 / 20 degrees',60,750,550,110,'gray','box')], [
('source','views','R L','',[]),('views','lrpair','R L','',[]),('lrpair','vosr','R L','',[]),
('vosr','controls','B T','cached native pair',[]),('controls','match','L R','',[]),
('match','auc','L R','estimated H',[]),
('views','auc','B T','H is evaluation-only',[(585,410),(285,410)]),
('controls','iqa','B T','paired quality',[]),('auc','extend','B T','planned; not yet implemented',[])],
['No dataset-name dependency','No multi-seed ranking or parameter sweep','Planar success does not establish 3D reconstruction success'])]


def anchor(node, side):
    _,_,x,y,w,h,_,_=node
    return {'L':(x,y+h/2),'R':(x+w,y+h/2),'T':(x+w/2,y),'B':(x+w/2,y+h)}[side]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    mx = ET.Element('mxfile',host='drawio',pages=str(len(PAGES)))
    for p in PAGES:
        diagram=ET.SubElement(mx,'diagram',id=p['name'],name=p['name'])
        model=ET.SubElement(diagram,'mxGraphModel',page='1',pageWidth='1560',pageHeight='1000',grid='1',gridSize='10')
        root=ET.SubElement(model,'root')
        ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
        svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1560" height="1000" viewBox="0 0 1560 1000">',
             '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#596779"/></marker></defs>',
             '<rect width="1560" height="1000" fill="white"/>']
        def label(id,text,x,y,w,h,size=18,color='#263648',align='center'):
            cell=ET.SubElement(root,'mxCell',id=id,value=text,parent='1',vertex='1',
                               style=f'text;html=0;fontSize={size};fontFamily=Arial;fontColor={color};align={align};')
            ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
            lines=text.split('\n')
            for i,line in enumerate(lines):
                svg.append(f'<text x="{x+w/2}" y="{y+h/2+(i-(len(lines)-1)/2)*(size+9)}" text-anchor="middle" dominant-baseline="middle" font-family="Arial,sans-serif" font-size="{size}" fill="{color}">{html.escape(line)}</text>')
        label('title',p['title'],40,30,1480,45,30)
        label('subtitle',p['subtitle'],40,90,1480,35,17,'#667587')
        nodes={n[0]:n for n in p['nodes']}
        # Edges first in SVG; all paths have explicit routing, shared with draw.io.
        for i,(src,dst,ports,text,points) in enumerate(p['edges']):
            start,end=ports.split(); sx,sy={'L':(0,.5),'R':(1,.5),'T':(.5,0),'B':(.5,1)}[start];ex,ey={'L':(0,.5),'R':(1,.5),'T':(.5,0),'B':(.5,1)}[end]
            style=f'edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;endArrow=block;strokeColor=#596779;strokeWidth=2;fontFamily=Arial;fontSize=14;labelBackgroundColor=#ffffff;exitX={sx};exitY={sy};entryX={ex};entryY={ey};'
            cell=ET.SubElement(root,'mxCell',id=f'edge-{i}',parent='1',edge='1',source=src,target=dst,value=text,style=style)
            geo=ET.SubElement(cell,'mxGeometry',relative='1',attrib={'as':'geometry'})
            if points:
                arr=ET.SubElement(geo,'Array',attrib={'as':'points'})
                for x,y in points:ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
            path=[anchor(nodes[src],start),*points,anchor(nodes[dst],end)]
            svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in path)+'" fill="none" stroke="#596779" stroke-width="2" marker-end="url(#arr)"/>')
            if text:
                lengths=[abs(a[0]-b[0])+abs(a[1]-b[1]) for a,b in zip(path[:-1],path[1:])]
                horizontal=[k for k,(a,b) in enumerate(zip(path[:-1],path[1:])) if a[1]==b[1]]
                k=max(horizontal,key=lambda k:lengths[k]) if horizontal else (len(path)-1)//2
                a,b=path[k:k+2];x,y=(a[0]+b[0])/2,(a[1]+b[1])/2-12
                geo.set('x',str(2*(sum(lengths[:k])+lengths[k]/2)/sum(lengths)-1))
                geo.set('y','-12')
                svg.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Arial" font-size="14" fill="#596779" stroke="white" stroke-width="5" paint-order="stroke">{html.escape(text)}</text>')
        for id,text,x,y,w,h,color,shape in p['nodes']:
            fill,stroke=PALETTE[color]
            style=SHAPES[shape]+f'whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontFamily=Arial;fontSize=18;spacing=14;'
            cell=ET.SubElement(root,'mxCell',id=id,parent='1',vertex='1',value=text,style=style)
            ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
            if shape=='decision':
                svg.append(f'<polygon points="{x+w/2},{y} {x+w},{y+h/2} {x+w/2},{y+h} {x},{y+h/2}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            elif shape=='ellipse':
                svg.append(f'<ellipse cx="{x+w/2}" cy="{y+h/2}" rx="{w/2}" ry="{h/2}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            elif shape=='input':
                svg.append(f'<polygon points="{x+18},{y} {x+w},{y} {x+w-18},{y+h} {x},{y+h}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            else:
                svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{12 if shape=="box" else 0}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
                if shape=='tensor':svg.append(f'<path d="M {x} {y+14} H {x+w-14} V {y+h} M {x+w-14} {y+14} L {x+w} {y}" fill="none" stroke="{stroke}"/>')
            lines=text.split('\n')
            for i,line in enumerate(lines):
                svg.append(f'<text x="{x+w/2}" y="{y+h/2+(i-(len(lines)-1)/2)*28}" text-anchor="middle" dominant-baseline="middle" font-family="Arial,sans-serif" font-size="18" fill="#263648">{html.escape(line)}</text>')
        label('footer','    |    '.join(p['notes']),40,945,1480,30,15,'#667587')
        svg.append('</svg>')
        (OUT/(p['name']+'.svg')).write_bytes('\n'.join(svg).encode('utf-8'))
    ET.indent(mx)
    (OUT/'vosr-v2.5.drawio').write_bytes(ET.tostring(mx,encoding='utf-8',xml_declaration=True).replace(b'\r\n',b'\n'))
    print('Generated three editable pages and same-layout SVG previews')


if __name__=='__main__':main()
