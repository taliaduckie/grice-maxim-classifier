"""Bake data/derived/natural_qa_pairs.csv into a standalone annotation page.

    python build_annotator.py && open annotator.html

Keyboard-driven, autosaves to localStorage per annotator, exports the same
CSV schema with maxim / violation_type / confidence / notes filled in.
Give annotator 2 the same annotator.html file; different initials keep
their labels separate, and their export is a separate CSV for agreement.py.
"""
import csv, json, os

SRC = os.path.join("data", "derived", "natural_qa_pairs.csv")
OUT = "annotator.html"

with open(SRC, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
cols = list(rows[0].keys())

payload = json.dumps({"cols": cols, "rows": rows}, ensure_ascii=False)
payload = payload.replace("</", "<\\/")  # never terminate the script tag

TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Gricean annotator</title>
<style>
 body{font:15px/1.45 -apple-system,system-ui,sans-serif;margin:0;background:#f4f2ee;color:#222}
 .wrap{max-width:880px;margin:0 auto;padding:18px}
 .top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
 .top h1{font-size:17px;margin:0}
 .prog{font-variant-numeric:tabular-nums;color:#555}
 .strip{display:grid;grid-template-columns:repeat(47,1fr);gap:2px;margin:10px 0}
 .strip div{height:9px;border-radius:2px;background:#d8d4cc;cursor:pointer}
 .strip div.done{background:#4a8f5c}.strip div.cur{outline:2px solid #1a66d0;outline-offset:1px}
 .card{background:#fff;border:1px solid #ddd;border-radius:10px;padding:16px 18px;margin:8px 0}
 .meta{font-size:12px;color:#777;display:flex;gap:10px;margin-bottom:8px}
 .meta .warn{color:#b3541e;font-weight:600}
 .ctx{color:#444;background:#faf8f4;border-left:3px solid #c9b98a;padding:8px 12px;border-radius:4px;white-space:pre-wrap}
 .utt{margin-top:10px;padding:8px 12px;border-left:3px solid #6b93c9;border-radius:4px;white-space:pre-wrap}
 .row{margin:10px 0 2px;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:#888}
 .btns{display:flex;gap:6px;flex-wrap:wrap}
 button{font:inherit;padding:6px 12px;border:1px solid #ccc;border-radius:7px;background:#fff;cursor:pointer}
 button.sel{background:#1a66d0;border-color:#1a66d0;color:#fff}
 button.dim{opacity:.4;pointer-events:none}
 button kbd{font-size:11px;color:#999;margin-right:5px}button.sel kbd{color:#cfe0fa}
 textarea{width:100%;box-sizing:border-box;font:inherit;border:1px solid #ccc;border-radius:7px;padding:7px;min-height:38px}
 .nav{display:flex;gap:8px;align-items:center;margin-top:12px;flex-wrap:wrap}
 .nav .spacer{flex:1}
 .hint{font-size:12px;color:#888;margin-top:10px}
 input[type=text]{font:inherit;width:70px;padding:5px 8px;border:1px solid #ccc;border-radius:7px}
 label{font-size:13px;color:#555}
</style></head><body><div class="wrap">
 <div class="top">
  <h1>Natural QA pairs — Gricean annotation</h1>
  <span class="prog" id="prog"></span>
  <span class="spacer" style="flex:1"></span>
  <label>annotator <input type="text" id="who" placeholder="tbh" maxlength="12"></label>
  <label><input type="checkbox" id="auto" checked> auto-advance</label>
 </div>
 <div class="strip" id="strip"></div>
 <div class="card">
  <div class="meta" id="meta"></div>
  <div class="ctx" id="ctx"></div>
  <div class="utt" id="utt"></div>
 </div>
 <div class="row">maxim</div><div class="btns" id="maxims"></div>
 <div class="row">violation type</div><div class="btns" id="vtypes"></div>
 <div class="row">confidence</div><div class="btns" id="confs"></div>
 <div class="row">notes <span style="text-transform:none">(press N to focus, Esc to leave)</span></div>
 <textarea id="notes" rows="2"></textarea>
 <div class="nav">
  <button id="prev">&larr; prev</button><button id="next">next &rarr;</button>
  <button id="skipnext">next unlabeled &darr;</button>
  <span class="spacer"></span>
  <button id="export">Export CSV</button>
  <button id="restore">Restore from exported CSV</button>
  <input type="file" id="file" accept=".csv" hidden>
 </div>
 <div class="hint">Keys: 1&ndash;5 maxim &middot; F/V/O/U violation &middot; Q/W/E/R/T confidence 1&ndash;5 &middot; arrows navigate &middot; Enter = next unlabeled. Everything autosaves per annotator; export whenever.</div>
</div>
<script>
const DATA = __DATA__;
const MAXIMS=["Cooperative","Quality","Quantity","Relation","Manner"];
const VTYPES=[["f","flouting"],["v","violating"],["o","opting_out"],["u","unknown"]];
const rows=DATA.rows; let i=0;
const $=id=>document.getElementById(id);
const who=()=>($("who").value.trim()||"anon");
const KEY=()=>"grice_annot_"+who();
let ann={};
function load(){try{ann=JSON.parse(localStorage.getItem(KEY()))||{}}catch(e){ann={}}}
function save(){try{localStorage.setItem(KEY(),JSON.stringify(ann))}catch(e){}}
const get=r=>ann[r.row_id]||{};
function set(r,k,v){ann[r.row_id]=Object.assign({},get(r),{[k]:v});
 if(k==="maxim"&&v==="Cooperative")ann[r.row_id].violation_type="none";
 if(k==="maxim"&&v!=="Cooperative"&&get(r).violation_type==="none")delete ann[r.row_id].violation_type;
 save();render();
 if($("auto").checked&&done(r)&&(k==="confidence"))nextUnlabeled();}
const done=r=>{const a=get(r);return !!(a.maxim&&a.violation_type&&a.confidence)};
function mkbtns(el,items,field){el.innerHTML="";items.forEach(([k,v])=>{
 const b=document.createElement("button");b.innerHTML="<kbd>"+k.toUpperCase()+"</kbd>"+v;
 b.onclick=()=>set(rows[i],field,v);b.dataset.v=v;el.appendChild(b);});}
function render(){
 const r=rows[i],a=get(r);
 const n=rows.filter(done).length;
 $("prog").textContent=n+" / "+rows.length+" labeled";
 $("meta").innerHTML="<span>#"+(i+1)+"</span><span>r/"+r.subreddit+"</span><span>"+r.pairing+"</span>"+
   (r.seen_in_training==="yes"?"<span class='warn'>seen in training</span>":"");
 $("ctx").textContent=r.context; $("utt").textContent=r.utterance;
 for(const [el,field] of [[ "maxims","maxim"],["vtypes","violation_type"],["confs","confidence"]])
   [...$(el).children].forEach(b=>{b.classList.toggle("sel",a[field]===b.dataset.v);
     if(el==="vtypes")b.classList.toggle("dim",a.maxim==="Cooperative");});
 if(document.activeElement!==$("notes"))$("notes").value=a.notes||"";
 [...$("strip").children].forEach((d,j)=>{d.classList.toggle("done",done(rows[j]));d.classList.toggle("cur",j===i);});
}
function go(j){i=(j+rows.length)%rows.length;render();window.scrollTo(0,0);}
function nextUnlabeled(){for(let s=1;s<=rows.length;s++){const j=(i+s)%rows.length;if(!done(rows[j])){go(j);return}}render();}
mkbtns($("maxims"),MAXIMS.map((m,k)=>[String(k+1),m]),"maxim");
mkbtns($("vtypes"),VTYPES,"violation_type");
mkbtns($("confs"),[["q","1"],["w","2"],["e","3"],["r","4"],["t","5"]],"confidence");
rows.forEach((r,j)=>{const d=document.createElement("div");d.onclick=()=>go(j);d.title="#"+(j+1);$("strip").appendChild(d);});
$("notes").oninput=e=>{set(rows[i],"notes",e.target.value)};
$("prev").onclick=()=>go(i-1);$("next").onclick=()=>go(i+1);$("skipnext").onclick=nextUnlabeled;
$("who").value=localStorage.getItem("grice_annot_who")||"";
$("who").oninput=()=>{localStorage.setItem("grice_annot_who",$("who").value);load();render();};
$("who").onkeydown=e=>{if(e.key==="Enter")e.target.blur()};
document.addEventListener("keydown",e=>{
 if(e.target===$("notes")){if(e.key==="Escape")$("notes").blur();return}
 if(e.target.tagName==="INPUT")return;
 const r=rows[i],k=e.key.toLowerCase();
 if(k>="1"&&k<="5")set(r,"maxim",MAXIMS[k-1]);
 else if(get(r).maxim!=="Cooperative"&&VTYPES.some(v=>v[0]===k))set(r,"violation_type",VTYPES.find(v=>v[0]===k)[1]);
 else if("qwert".includes(k)&&k)set(r,"confidence",String("qwert".indexOf(k)+1));
 else if(e.key==="ArrowRight")go(i+1);else if(e.key==="ArrowLeft")go(i-1);
 else if(e.key==="Enter")nextUnlabeled();else if(k==="n"){e.preventDefault();$("notes").focus()}
});
function csvField(s){s=String(s==null?"":s);return /[",\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s}
$("export").onclick=()=>{
 const cols=DATA.cols.concat(["annotator"]);
 const out=[cols.map(csvField).join(",")];
 rows.forEach(r=>{const a=get(r),c=Object.assign({},r);
  c.maxim=a.maxim||"";c.violation_type=a.violation_type||"";
  c.confidence_1_to_5=a.confidence||"";c.notes=a.notes||"";c.annotator=who();
  out.push(cols.map(k=>csvField(c[k])).join(","));});
 const blob=new Blob([out.join("\n")+"\n"],{type:"text/csv"});
 const u=URL.createObjectURL(blob),el=document.createElement("a");
 el.href=u;el.download="natural_qa_annotated_"+who()+".csv";el.click();URL.revokeObjectURL(u);};
$("restore").onclick=()=>$("file").click();
$("file").onchange=e=>{const f=e.target.files[0];if(!f)return;
 f.text().then(t=>{const recs=parseCSV(t);if(!recs.length)return alert("no rows parsed");
  const hdr=recs[0],idx=Object.fromEntries(hdr.map((h,j)=>[h,j]));
  let n=0;recs.slice(1).forEach(rec=>{const id=rec[idx.row_id];if(!id)return;
   const a={};["maxim","violation_type","notes"].forEach(k=>{if(rec[idx[k]])a[k]=rec[idx[k]]});
   if(rec[idx.confidence_1_to_5])a.confidence=rec[idx.confidence_1_to_5];
   if(Object.keys(a).length){ann[id]=a;n++}});
  save();render();alert("restored labels for "+n+" rows into annotator '"+who()+"'");});};
function parseCSV(t){const out=[[]];let f="",q=false;
 for(let c=0;c<t.length;c++){const ch=t[c];
  if(q){if(ch==='"'){if(t[c+1]==='"'){f+='"';c++}else q=false}else f+=ch}
  else if(ch==='"')q=true;
  else if(ch===","){out.at(-1).push(f);f=""}
  else if(ch==="\n"||ch==="\r"){if(ch==="\r"&&t[c+1]==="\n")c++;out.at(-1).push(f);f="";out.push([])}
  else f+=ch}
 if(f!==""||out.at(-1).length)out.at(-1).push(f);
 return out.filter(r=>r.length>1||r[0]!=="");}
load();render();
</script></body></html>
"""

html = TEMPLATE.replace("__DATA__", payload)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {OUT}: {len(rows)} pairs, {os.path.getsize(OUT)//1024}KB")
