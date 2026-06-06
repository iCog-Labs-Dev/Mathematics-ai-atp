from __future__ import annotations

import html
import json
import os
import tempfile
import webbrowser
from pathlib import Path

from .graph import DAGBuilder


TYPE_COLORS = {
    "var": {"fill": "#dbeafe", "stroke": "#3b82f6"},
    "type": {"fill": "#fee2e2", "stroke": "#ef4444"},
    "predicate": {"fill": "#fef3c7", "stroke": "#f59e0b"},
    "operator": {"fill": "#dcfce7", "stroke": "#22c55e"},
    "app": {"fill": "#ede9fe", "stroke": "#8b5cf6"},
    "meta": {"fill": "#f3f4f6", "stroke": "#6b7280"},
}


def _short_label(label: str, max_length: int = 13) -> str:
    if len(label) <= max_length:
        return label
    if max_length <= 3:
        return label[:max_length]
    return f"{label[:max_length - 3]}..."


def build_visualization_html(
    dag: DAGBuilder,
    *,
    title: str = "Lean Proof State DAG",
    theorem: str = "",
    tactic: str = "",
) -> str:
    parent_uses = dag.outgoing_counts()
    child_counts = dag.incoming_counts()
    stats = dag.stats().as_dict()

    nodes_json = []
    for node in dag.nodes:
        colors = TYPE_COLORS.get(node.node_type, TYPE_COLORS["meta"])
        nodes_json.append(
            {
                "id": node.id,
                "label": node.label,
                "display": _short_label(node.label),
                "type": node.node_type,
                "parent_uses": parent_uses[node.id],
                "children": child_counts[node.id],
                "shared": parent_uses[node.id] > 1,
                "fill": colors["fill"],
                "stroke": colors["stroke"],
            }
        )

    links_json = [{"source": source, "target": target} for (source, target) in dag.edges]
    safe_title = html.escape(title)
    safe_theorem = html.escape(theorem)
    tactic_display = tactic[:70] + "..." if len(tactic) > 70 else tactic
    safe_tactic = html.escape(tactic_display)

    html_doc = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0f1117;color:#e2e8f0;height:100vh;display:flex;flex-direction:column}
#header{padding:14px 20px;border-bottom:1px solid #2d3748;flex-shrink:0}
#header h1{font-size:14px;font-weight:500;color:#f8fafc;margin-bottom:3px}
#header .meta{font-size:11px;color:#64748b;font-family:monospace}
#header .meta span{margin-right:20px}
#stats{display:flex;border-bottom:1px solid #2d3748;flex-shrink:0}
.stat{padding:8px 20px;border-right:1px solid #2d3748;font-size:12px}
.stat .val{font-size:18px;font-weight:600;color:#f8fafc;display:block}
.stat .lab{color:#64748b}
#main{display:flex;flex:1;min-height:0}
#svg-wrap{flex:1;overflow:hidden;position:relative}
svg{width:100%;height:100%}
#sidebar{width:270px;border-left:1px solid #2d3748;padding:14px;
         overflow-y:auto;display:flex;flex-direction:column;gap:10px;flex-shrink:0}
.scard{background:#1e2433;border:1px solid #2d3748;border-radius:8px;padding:12px}
.scard h3{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em;
          margin-bottom:8px;font-weight:500}
#info-box{min-height:90px;font-size:12px;color:#94a3b8;line-height:1.7}
#info-box .iname{font-size:15px;font-weight:600;color:#f8fafc;
                font-family:monospace;margin-bottom:4px;word-break:break-all}
#info-box .irow{display:flex;justify-content:space-between;
               border-top:1px solid #2d3748;padding-top:5px;
               margin-top:5px;font-size:11px}
#info-box .ik{color:#64748b}
.shared-badge{display:inline-block;background:#854f0b;color:#fef3c7;
              font-size:10px;padding:2px 7px;border-radius:10px;margin-top:3px}
.leg-row{display:flex;align-items:center;gap:7px;font-size:12px;
         color:#94a3b8;margin-bottom:4px}
.leg-dot{width:11px;height:11px;border-radius:50%;border:2px solid;flex-shrink:0}
.ctrl-row{margin-bottom:7px;font-size:12px;color:#94a3b8}
label{display:flex;align-items:center;gap:6px;cursor:pointer}
input[type=range]{width:100%;margin-top:3px;accent-color:#8b5cf6}
button{background:#2d3748;border:1px solid #4a5568;color:#e2e8f0;
       padding:6px 10px;border-radius:6px;cursor:pointer;font-size:12px;
       width:100%;margin-top:4px}
button:hover{background:#374151}
.node{cursor:pointer}
</style>
</head>
<body>
<div id="header">
  <h1>__TITLE__</h1>
  <div class="meta">__META__</div>
</div>
<div id="stats">
  <div class="stat"><span class="val">__STAT_NODES__</span><span class="lab">nodes</span></div>
  <div class="stat"><span class="val">__STAT_EDGES__</span><span class="lab">edges</span></div>
  <div class="stat"><span class="val">__STAT_REUSED__</span><span class="lab">reused nodes</span></div>
  <div class="stat"><span class="val">__STAT_RATIO__</span><span class="lab">sharing ratio</span></div>
</div>
<div id="main">
  <div id="svg-wrap"><svg id="graph"></svg></div>
  <div id="sidebar">
    <div class="scard"><h3>Selected node</h3><div id="info-box">Click any node to inspect it</div></div>
    <div class="scard">
      <h3>Node types</h3>
      <div class="leg-row"><div class="leg-dot" style="background:#dbeafe;border-color:#3b82f6"></div>variable (n, m, x...)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#fee2e2;border-color:#ef4444"></div>type (Nat, Type...)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#fef3c7;border-color:#f59e0b"></div>predicate (Even, Ring...)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#dcfce7;border-color:#22c55e"></div>operator (+, =, ->...)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#ede9fe;border-color:#8b5cf6"></div>App (function call)</div>
      <div class="leg-row"><div class="leg-dot" style="background:#f3f4f6;border-color:#6b7280"></div>Hyp / Goal / State</div>
      <div class="leg-row">
        <svg width="13" height="13" style="flex-shrink:0">
          <circle cx="6.5" cy="6.5" r="5" fill="none" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="2,1.5"/>
        </svg>
        dashed ring = reused subexpression
      </div>
    </div>
    <div class="scard">
      <h3>Layout controls</h3>
      <div class="ctrl-row"><label>Link distance<input type="range" id="sl-dist" min="20" max="200" value="70"></label></div>
      <div class="ctrl-row"><label>Repulsion<input type="range" id="sl-charge" min="-800" max="-20" value="-180"></label></div>
      <div class="ctrl-row"><label><input type="checkbox" id="cb-labels" checked> Show labels</label></div>
      <div class="ctrl-row"><label><input type="checkbox" id="cb-shared"> Highlight reused only</label></div>
      <button id="btn-reset">Reset layout</button>
    </div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
const NODES=__NODES__;
const LINKS=__LINKS__;

const svg=d3.select('#graph');
const wrap=document.getElementById('svg-wrap');
const W=()=>wrap.clientWidth, H=()=>wrap.clientHeight;

const g=svg.append('g');
svg.call(d3.zoom().scaleExtent([0.05,8]).on('zoom',e=>g.attr('transform',e.transform)));

svg.append('defs').append('marker')
  .attr('id','arr').attr('viewBox','0 0 10 10')
  .attr('refX',9).attr('refY',5).attr('markerWidth',5).attr('markerHeight',5)
  .attr('orient','auto-start-reverse')
  .append('path').attr('d','M2 1L8 5L2 9')
  .attr('fill','none').attr('stroke','context-stroke')
  .attr('stroke-width',1.5).attr('stroke-linecap','round').attr('stroke-linejoin','round');

const sim=d3.forceSimulation(NODES)
  .force('link',d3.forceLink(LINKS).id(d=>d.id).distance(70).strength(0.3))
  .force('charge',d3.forceManyBody().strength(-180))
  .force('center',d3.forceCenter(W()/2,H()/2))
  .force('collide',d3.forceCollide(28));

const linkSel=g.append('g').selectAll('line').data(LINKS).join('line')
  .attr('stroke','rgba(148,163,184,0.28)').attr('stroke-width',1.5)
  .attr('marker-end','url(#arr)');

const nodeR=d=>d.label==='State'?26:d.type==='meta'?21:17;
const nodeSel=g.append('g').selectAll('g').data(NODES).join('g').attr('class','node');

nodeSel.append('circle')
  .attr('r',nodeR).attr('fill',d=>d.fill).attr('stroke',d=>d.stroke)
  .attr('stroke-width',d=>d.shared?2.5:1.5);

nodeSel.filter(d=>d.shared).append('circle')
  .attr('r',d=>nodeR(d)+7).attr('fill','none').attr('stroke',d=>d.stroke)
  .attr('stroke-width',1).attr('stroke-dasharray','3,2').attr('opacity',0.65)
  .attr('pointer-events','none');

const fs=d=>d.display.length>=10?'7.5px':d.display.length>=7?'9px':'11px';
nodeSel.append('text')
  .attr('text-anchor','middle').attr('dominant-baseline','central')
  .attr('font-size',fs).attr('font-weight','500').attr('fill','#1e293b')
  .attr('pointer-events','none').text(d=>d.display);

nodeSel.call(d3.drag()
  .on('start',(e,d)=>{if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y;})
  .on('drag', (e,d)=>{d.fx=e.x;d.fy=e.y;})
  .on('end',  (e,d)=>{if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null;}));

let sel=null;
function resetHL(){
  nodeSel.selectAll('circle,text').attr('opacity',1);
  linkSel.attr('stroke-opacity',0.7).attr('stroke','rgba(148,163,184,0.28)').attr('stroke-width',1.5);
  sel=null;
}

nodeSel.on('click',(e,d)=>{
  e.stopPropagation(); sel=d.id;
  const parents=new Set(), children=new Set(), parentLinks=new Set(), childLinks=new Set();
  LINKS.forEach(l=>{
    const s=l.source.id!==undefined?l.source.id:l.source;
    const t=l.target.id!==undefined?l.target.id:l.target;
    if(s===d.id){parents.add(t);parentLinks.add(l);}
    if(t===d.id){children.add(s);childLinks.add(l);}
  });
  nodeSel.selectAll('circle,text').attr('opacity',nd=>nd.id===d.id||parents.has(nd.id)||children.has(nd.id)?1:0.08);
  linkSel
    .attr('stroke-opacity',l=>childLinks.has(l)||parentLinks.has(l)?1:0.03)
    .attr('stroke',l=>childLinks.has(l)?'#f59e0b':parentLinks.has(l)?d.stroke:'rgba(148,163,184,0.28)')
    .attr('stroke-width',l=>childLinks.has(l)||parentLinks.has(l)?2.5:1.5);
  const parentCount=parentLinks.size, childCount=childLinks.size;
  document.getElementById('info-box').innerHTML=
    '<div class="iname">'+d.label+'</div>'+
    '<div style="font-size:11px;color:#64748b;margin-bottom:4px">type: <b style="color:#e2e8f0">'+d.type+'</b></div>'+
    (d.shared?'<div class="shared-badge">reused by '+parentCount+' parent'+(parentCount!==1?'s':'')+'</div>':'')+
    '<div class="irow"><span class="ik">node id</span><span>'+d.id+'</span></div>'+
    '<div class="irow"><span class="ik">parents (uses)</span><span>'+parentCount+'</span></div>'+
    '<div class="irow"><span class="ik">children (inputs)</span><span>'+childCount+'</span></div>'+
    (parentCount>1?'<div style="margin-top:8px;font-size:11px;color:#64748b">Without DAG sharing this subexpression would need '+parentCount+' copies.</div>':'');
});

svg.on('click',()=>{
  if(sel!==null){resetHL();document.getElementById('info-box').textContent='Click any node to inspect it';}
});

function ex(l,which){
  const s=l.source,t=l.target,r=nodeR(which==='s'?s:t)+(which==='s'?0:7);
  const dx=t.x-s.x,dy=t.y-s.y,dist=Math.sqrt(dx*dx+dy*dy)||1;
  return which==='s'?{x:s.x+dx/dist*r,y:s.y+dy/dist*r}:{x:t.x-dx/dist*r,y:t.y-dy/dist*r};
}

sim.on('tick',()=>{
  linkSel
    .attr('x1',l=>ex(l,'s').x).attr('y1',l=>ex(l,'s').y)
    .attr('x2',l=>ex(l,'t').x).attr('y2',l=>ex(l,'t').y);
  nodeSel.attr('transform',d=>'translate('+d.x+','+d.y+')');
});

document.getElementById('sl-dist').oninput=function(){sim.force('link').distance(+this.value);sim.alpha(0.4).restart();};
document.getElementById('sl-charge').oninput=function(){sim.force('charge').strength(+this.value);sim.alpha(0.4).restart();};
document.getElementById('cb-labels').onchange=function(){nodeSel.selectAll('text').attr('display',this.checked?null:'none');};
document.getElementById('cb-shared').onchange=function(){
  if(this.checked){
    nodeSel.selectAll('circle,text').attr('opacity',d=>d.shared?1:0.12);
    linkSel.attr('stroke-opacity',l=>{
      const s=l.source.id!==undefined?l.source.id:l.source;
      const t=l.target.id!==undefined?l.target.id:l.target;
      const sn=NODES.find(n=>n.id===s),tn=NODES.find(n=>n.id===t);
      return (sn&&sn.shared)||(tn&&tn.shared)?0.8:0.04;
    });
  }else{nodeSel.selectAll('circle,text').attr('opacity',1);linkSel.attr('stroke-opacity',0.7);}
};
document.getElementById('btn-reset').onclick=()=>{
  NODES.forEach(n=>{n.x=undefined;n.y=undefined;n.fx=null;n.fy=null;n.vx=0;n.vy=0;});
  sim.force('center',d3.forceCenter(W()/2,H()/2)).alpha(1).restart();
  resetHL();
  document.getElementById('info-box').textContent='Click any node to inspect it';
  document.getElementById('cb-shared').checked=false;
  nodeSel.selectAll('circle,text').attr('opacity',1);
  linkSel.attr('stroke-opacity',0.7);
};
</script>
</body>
</html>"""

    meta_parts = []
    if theorem:
        meta_parts.append(f'<span>theorem: <b style="color:#a5b4fc">{safe_theorem}</b></span>')
    if tactic:
        meta_parts.append(f'<span>tactic: <b style="color:#6ee7b7">{safe_tactic}</b></span>')

    replacements = {
        "__TITLE__": safe_title,
        "__META__": "".join(meta_parts),
        "__STAT_NODES__": str(stats["num_nodes"]),
        "__STAT_EDGES__": str(stats["num_edges"]),
        "__STAT_REUSED__": str(stats["num_reused_nodes"]),
        "__STAT_RATIO__": f"{stats['sharing_ratio']:.2f}",
        "__NODES__": json.dumps(nodes_json),
        "__LINKS__": json.dumps(links_json),
    }

    for placeholder, value in replacements.items():
        html_doc = html_doc.replace(placeholder, value)
    return html_doc


def visualize_dag(
    dag: DAGBuilder,
    *,
    title: str = "Lean Proof State DAG",
    theorem: str = "",
    tactic: str = "",
    open_browser: bool = True,
    output_path: str | None = None,
) -> str:
    html_doc = build_visualization_html(dag, title=title, theorem=theorem, tactic=tactic)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8")
        tmp.write(html_doc)
        tmp.close()
        resolved_output = tmp.name
    else:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html_doc, encoding="utf-8")
        resolved_output = str(output)

    if open_browser:
        webbrowser.open("file://" + os.path.abspath(resolved_output))

    return resolved_output
