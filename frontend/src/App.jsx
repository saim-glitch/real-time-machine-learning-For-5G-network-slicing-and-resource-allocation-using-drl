import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import { LineChart, Line, BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
         XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, Cell, AreaChart, Area,
         ComposedChart, ReferenceLine } from "recharts";
import { Activity, Cpu, Clock, Shield, Radio, Zap, BarChart3, Network, FlaskConical,
         Play, Pause, RotateCcw, Gauge, TrendingUp, AlertTriangle, CheckCircle, Wifi,
         ChevronLeft, ChevronRight, Layers, Brain, ArrowUpRight, ArrowDownRight, Minus } from "lucide-react";

// ═══════════════════ THEME ═══════════════════
const C = {
  bg: "#060A14", card: "rgba(12,20,40,0.85)", border: "rgba(255,255,255,0.07)",
  text: "#E2E8F0", dim: "#64748B", embb: "#00D4FF", urllc: "#FF006E",
  mmtc: "#39FF14", ok: "#22C55E", warn: "#F59E0B", bad: "#EF4444",
  grid: "rgba(255,255,255,0.04)", accent: "#7C3AED", purple: "#A855F7",
};
const SC = [C.embb, C.urllc, C.mmtc];
const SN = ["eMBB", "URLLC", "mMTC"];
const ttStyle = { background: "rgba(8,14,30,0.96)", border: `1px solid ${C.border}`, borderRadius: 10, fontSize: 12, color: C.text, backdropFilter: "blur(12px)" };

// ═══════════════════ SIMULATION ENGINE ═══════════════════
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, t) => a + (b - a) * t;
const rn = () => { let u=0,v=0; while(!u)u=Math.random(); while(!v)v=Math.random(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); };

function createSim() {
  let al = [33,33,34], tp = [80,8,12], lt = [35,1.2,400], dr = [.015,.0005,.06], au = [25,10,160];
  let rw = 0.3, step = 0, mmpp = 0, tm = [1,1,1], lim = 100, eps = 1.0;
  const tick = () => {
    step++;
    if (mmpp===0 && Math.random()<.003) mmpp=1;
    if (mmpp===1 && Math.random()<.006) mmpp=0;
    const lf = Math.min(1, step/3000);
    eps = Math.max(0.01, 1.0 - step/8000);
    const urg = [tp[0]<100?.3:-.1, lt[1]>.8?.5:-.05, au[2]<180?.2:-.05];
    for(let k=0;k<3;k++){const d=(urg[k]*lf+(Math.random()-.5)*(1-lf*.8))*3; al[k]=clamp(Math.round(al[k]+d),[10,5,5][k],[60,30,40][k]);}
    const tot=al[0]+al[1]+al[2]; if(tot>lim){let ex=tot-lim; al[2]=Math.max(5,al[2]-Math.ceil(ex/2)); al[0]=Math.max(10,al[0]-Math.floor(ex/2));}
    const tl=mmpp===1?1.5:.8;
    tp[0]=lerp(tp[0],clamp(al[0]*2.8*(.85+rn()*.08)*tl*tm[0],20,180),.08);
    tp[1]=lerp(tp[1],clamp(al[1]*1.2*(.9+rn()*.03)*tm[1],2,15),.08);
    tp[2]=lerp(tp[2],clamp(al[2]*.4*(.7+rn()*.1)*tm[2],3,30),.08);
    const qf=al.map((a,k)=>Math.max(.1,1-a/[50,25,30][k]));
    lt[0]=lerp(lt[0],clamp(12+qf[0]*45+rn()*5,5,95)*tm[0],.06);
    lt[1]=lerp(lt[1],clamp(.15+qf[1]*1.2+Math.abs(rn())*.08,.1,3)*tm[1],.06);
    lt[2]=lerp(lt[2],clamp(80+qf[2]*500+rn()*40,30,950)*tm[2],.06);
    dr[0]=lerp(dr[0],clamp(.002+qf[0]*.02+Math.abs(rn())*.003,0,.08),.05);
    dr[1]=lerp(dr[1],clamp(qf[1]*.0003+Math.abs(rn())*.00002,0,.005),.05);
    dr[2]=lerp(dr[2],clamp(.01+qf[2]*.06+Math.abs(rn())*.008,0,.15),.05);
    au[0]=Math.round(lerp(au[0],clamp(30*(1-dr[0]*3)*Math.min(1,al[0]/20),10,30),.04));
    au[1]=10;
    au[2]=Math.round(lerp(au[2],clamp(200*(1-dr[2])*Math.min(1,al[2]/12),80,200),.04));
    const Re=.6*Math.log(1+tp[0]/100)+.2*Math.max(0,1-lt[0]/100)+.2*(1-dr[0]);
    const Ru=.1*Math.min(tp[1]/10,1)+.5*Math.exp(-lt[1])+.4*(1-dr[1])**2;
    const Rm=.2*Math.min(tp[2]/20,1)+.2*Math.max(0,1-lt[2]/1000)+.6*Math.min(au[2]/200,1);
    rw=lerp(rw,.3*Re+.5*Ru+.2*Rm,.05);
    const ut=(al[0]+al[1]+al[2])/lim;
    const loss = Math.max(0.001, 0.5 * Math.exp(-step/2000) + Math.abs(rn()) * 0.05);
    return { step, al:[...al], tp:tp.map(t=>+t.toFixed(2)), lt:lt.map(l=>+l.toFixed(3)),
      dr:dr.map(d=>+d.toFixed(5)), au:[...au], rw:+rw.toFixed(4), ut:+(ut).toFixed(3),
      sla:lt[1]<1&&dr[1]<.001, mmpp, tt:+(tp[0]+tp[1]+tp[2]).toFixed(1), eps:+eps.toFixed(3), loss:+loss.toFixed(4) };
  };
  return { tick,
    scenario: (s)=>{if(s==="flash_crowd")tm[0]=4;else if(s==="urllc_emergency")tm[1]=6;else if(s==="mmtc_storm")tm[2]=5;else if(s==="network_degradation")lim=50;else{tm=[1,1,1];lim=100;}},
    reset:()=>{al=[33,33,34];tp=[80,8,12];lt=[35,1.2,400];dr=[.015,.0005,.06];au=[25,10,160];rw=.3;step=0;tm=[1,1,1];lim=100;eps=1;}
  };
}

// ═══════════════════ TRAINING DATA ═══════════════════
const ALGOS = [
  { n:"DQN", c:"#3B82F6", conv:350, fr:.71, v:.06 },
  { n:"DDQN", c:"#22C55E", conv:280, fr:.77, v:.045 },
  { n:"Dueling", c:"#F59E0B", conv:250, fr:.79, v:.04 },
  { n:"PPO", c:"#A855F7", conv:220, fr:.80, v:.03 },
  { n:"SAC", c:"#EF4444", conv:200, fr:.81, v:.025 },
];

function genTraining(){
  const d=[];
  for(let ep=1;ep<=500;ep++){
    const p={episode:ep};
    ALGOS.forEach(a=>{const pr=Math.min(1,ep/a.conv);const b=.25+(a.fr-.25)*(1-Math.exp(-3*pr));p[a.n]=+(b+(Math.random()-.5)*a.v*(1-pr*.7)).toFixed(4);
      p[a.n+"_loss"]=+(Math.max(0.01,0.8*Math.exp(-ep/a.conv*2)+(Math.random()-.5)*0.08)).toFixed(4);
      p[a.n+"_eps"]=+(Math.max(0.01,1-ep/350)).toFixed(3);
    });d.push(p);}return d;
}

function genCDF(){
  const d=[];
  const algos=["SAC","DDQN","Round Robin"];
  for(let x=0;x<=3;x+=0.05){
    const p={x:+x.toFixed(2)};
    p["SAC"]=+Math.min(1,1-Math.exp(-5*x)).toFixed(4);
    p["DDQN"]=+Math.min(1,1-Math.exp(-3.5*x)).toFixed(4);
    p["Round Robin"]=+Math.min(1,1-Math.exp(-1.2*x)).toFixed(4);
    d.push(p);
  }return d;
}

const COMP = [
  {n:"DQN",rw:.712,et:98.3,ul:.89,mc:178,sv:2.1,cv:350},
  {n:"DDQN",rw:.768,et:112.5,ul:.74,mc:185,sv:.8,cv:280},
  {n:"Dueling",rw:.789,et:118.7,ul:.71,mc:189,sv:.5,cv:250},
  {n:"PPO",rw:.801,et:121.3,ul:.68,mc:192,sv:.3,cv:220},
  {n:"SAC",rw:.812,et:125.1,ul:.65,mc:195,sv:.2,cv:200},
  {n:"Round Robin",rw:.534,et:76.2,ul:2.31,mc:145,sv:15.3,cv:null},
  {n:"Prop. Fair",rw:.623,et:89.4,ul:1.45,mc:162,sv:8.7,cv:null},
  {n:"Priority",rw:.589,et:68.1,ul:.92,mc:110,sv:5.2,cv:null},
];

// ═══════════════════ COMPONENTS ═══════════════════
const Glass = ({children,glow="",style={},...p}) => (
  <div style={{background:C.card,backdropFilter:"blur(20px)",border:`1px solid ${C.border}`,borderRadius:16,padding:20,
    boxShadow:glow?`0 0 30px ${glow}12, 0 8px 32px rgba(0,0,0,0.4)`:"0 8px 32px rgba(0,0,0,0.3)",transition:"all 0.3s",...style}} {...p}>{children}</div>
);

const Label = ({children}) => <p style={{color:C.dim,fontSize:11,letterSpacing:1.5,textTransform:"uppercase",marginBottom:12,fontWeight:500}}>{children}</p>;

const Num = ({v,u="",color=C.text,size=28}) => (
  <span style={{color,fontSize:size,fontWeight:700,letterSpacing:-1,fontFamily:"'JetBrains Mono',monospace"}}>
    {v}<span style={{fontSize:size*.45,color:C.dim,marginLeft:3}}>{u}</span>
  </span>
);

const Badge = ({name,color}) => (
  <span style={{background:`${color}18`,color,padding:"3px 10px",borderRadius:20,fontSize:11,fontWeight:600,border:`1px solid ${color}35`}}>{name}</span>
);

const Trend = ({val,good="up"}) => {
  const up = val > 0;
  const ok = (good==="up"&&up)||(good==="down"&&!up);
  const Icon = up ? ArrowUpRight : val < 0 ? ArrowDownRight : Minus;
  return <span style={{display:"inline-flex",alignItems:"center",gap:2,color:ok?C.ok:C.bad,fontSize:11}}><Icon size={12}/>{Math.abs(val).toFixed(1)}%</span>;
};

// ═══════════════════ ANIMATED SANKEY ═══════════════════
const SankeyFlow = ({allocation}) => {
  const total = allocation[0]+allocation[1]+allocation[2];
  const pcts = allocation.map(a=>a/total);
  let y = 20;
  const flows = pcts.map((p,k) => {
    const h = p * 160;
    const startY = y;
    y += h + 8;
    return { k, h, startY, endY: 30 + k * 70 };
  });
  
  return (
    <svg viewBox="0 0 300 220" style={{width:"100%",height:200}}>
      {/* Source bar */}
      <rect x="10" y="15" width="30" height="190" rx="6" fill={`${C.accent}30`} stroke={C.accent} strokeWidth="0.5" />
      <text x="25" y="12" textAnchor="middle" fill={C.dim} fontSize="9">{total} PRBs</text>
      
      {flows.map(({k,h,startY,endY}) => {
        const flowH = Math.max(h, 20);
        return (
          <g key={k}>
            {/* Flow path */}
            <path d={`M 40 ${startY} C 140 ${startY}, 160 ${endY}, 250 ${endY}`}
              fill="none" stroke={SC[k]} strokeWidth={flowH * 0.15 + 2} opacity="0.25" strokeLinecap="round" />
            <path d={`M 40 ${startY} C 140 ${startY}, 160 ${endY}, 250 ${endY}`}
              fill="none" stroke={SC[k]} strokeWidth={Math.max(1.5, flowH * 0.05)} opacity="0.8" strokeLinecap="round">
              <animate attributeName="stroke-dashoffset" from="400" to="0" dur={`${2+k*0.5}s`} repeatCount="indefinite" />
              <animate attributeName="stroke-dasharray" values="4 20;8 12;4 20" dur={`${2+k*0.3}s`} repeatCount="indefinite" />
            </path>
            {/* Target bar */}
            <rect x="250" y={endY-15} width="30" height="30" rx="6" fill={`${SC[k]}20`} stroke={SC[k]} strokeWidth="0.5" />
            <text x="295" y={endY-2} fill={SC[k]} fontSize="9" fontWeight="600">{SN[k]}</text>
            <text x="295" y={endY+10} fill={C.dim} fontSize="8">{allocation[k]} PRBs</text>
          </g>
        );
      })}
    </svg>
  );
};

// ═══════════════════ DASHBOARD PAGE ═══════════════════
function DashboardPage({m, hist, run, onStart, onStop, onReset, spd, setSpd}) {
  const tt = m.tp.reduce((a,b)=>a+b,0);
  const ut = m.ut*100;
  const sr = (1-(m.dr[0]+m.dr[1]+m.dr[2])/3)*100;

  // QoS Radar data
  const radarData = [
    {axis:"Throughput",eMBB:Math.min(m.tp[0]/130,1),URLLC:Math.min(m.tp[1]/12,1),mMTC:Math.min(m.tp[2]/25,1)},
    {axis:"Latency",eMBB:Math.max(0,1-m.lt[0]/100),URLLC:Math.max(0,1-m.lt[1]/2),mMTC:Math.max(0,1-m.lt[2]/1000)},
    {axis:"Reliability",eMBB:1-m.dr[0]*10,URLLC:1-m.dr[1]*1000,mMTC:1-m.dr[2]*5},
    {axis:"Efficiency",eMBB:m.ut,URLLC:m.ut,mMTC:m.ut},
    {axis:"Users",eMBB:m.au[0]/30,URLLC:m.au[1]/10,mMTC:m.au[2]/200},
  ];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      {/* Controls */}
      <div style={{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
        <button onClick={run?onStop:onStart} style={{background:run?`${C.bad}20`:`${C.ok}20`,border:`1px solid ${run?C.bad:C.ok}50`,color:run?C.bad:C.ok,borderRadius:10,padding:"8px 20px",cursor:"pointer",display:"flex",alignItems:"center",gap:6,fontSize:13,fontWeight:600}}>
          {run?<Pause size={14}/>:<Play size={14}/>}{run?"Pause":"Start Simulation"}
        </button>
        <button onClick={onReset} style={{background:"rgba(255,255,255,0.04)",border:`1px solid ${C.border}`,color:C.dim,borderRadius:10,padding:"8px 14px",cursor:"pointer",fontSize:13,display:"flex",alignItems:"center",gap:5}}>
          <RotateCcw size={13}/>Reset
        </button>
        <div style={{marginLeft:"auto",display:"flex",alignItems:"center",gap:6}}>
          <span style={{color:C.dim,fontSize:11}}>Speed:</span>
          {[1,2,5,10,20].map(s=>(
            <button key={s} onClick={()=>setSpd(s)} style={{background:spd===s?`${C.accent}35`:"rgba(255,255,255,0.03)",border:`1px solid ${spd===s?C.accent:C.border}`,color:spd===s?C.accent:C.dim,borderRadius:7,padding:"3px 10px",cursor:"pointer",fontSize:11,fontWeight:spd===s?600:400}}>{s}x</button>
          ))}
        </div>
        <div style={{display:"flex",alignItems:"center",gap:5}}>
          <div style={{width:7,height:7,borderRadius:"50%",background:run?C.ok:C.bad,boxShadow:run?`0 0 8px ${C.ok}`:"none"}}/>
          <span style={{color:C.dim,fontSize:11,fontFamily:"monospace"}}>Step {m.step.toLocaleString()}</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(200px,1fr))",gap:12}}>
        {[
          {icon:TrendingUp,label:"Total Throughput",val:tt.toFixed(1),unit:"Mbps",sub:<Trend val={12.5}/>,color:C.embb},
          {icon:Clock,label:"URLLC Latency",val:m.lt[1].toFixed(2),unit:"ms",sub:m.sla?<span style={{color:C.ok}}>✓ Under 1ms</span>:<span style={{color:C.bad}}>✗ SLA VIOLATED</span>,color:m.sla?C.ok:C.bad},
          {icon:Cpu,label:"Utilization",val:ut.toFixed(1),unit:"%",sub:"Target: 85%",color:C.accent},
          {icon:Shield,label:"SLA Compliance",val:sr.toFixed(2),unit:"%",sub:`${Math.max(0,Math.round((1-sr/100)*m.step))} violations`,color:C.ok},
        ].map(kpi=>(
          <Glass key={kpi.label} glow={kpi.color}>
            <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
              <div>
                <p style={{color:C.dim,fontSize:10,letterSpacing:1.5,textTransform:"uppercase",marginBottom:6}}>{kpi.label}</p>
                <Num v={kpi.val} u={kpi.unit} color={C.text}/>
                <p style={{color:C.dim,fontSize:11,marginTop:6}}>{kpi.sub}</p>
              </div>
              <div style={{background:`${kpi.color}15`,borderRadius:12,padding:10}}><kpi.icon size={20} color={kpi.color}/></div>
            </div>
          </Glass>
        ))}
      </div>

      {/* HERO: Resource Allocation Bar + Sankey */}
      <div style={{display:"grid",gridTemplateColumns:"2fr 1fr",gap:14}}>
        <Glass glow={C.accent}>
          <Label>Live PRB Allocation</Label>
          <div style={{display:"flex",height:52,borderRadius:10,overflow:"hidden",background:"rgba(0,0,0,0.35)",border:`1px solid ${C.border}`}}>
            {m.al.map((a,k)=>(
              <div key={k} style={{width:`${a}%`,background:`linear-gradient(135deg,${SC[k]}35,${SC[k]}10)`,borderRight:k<2?`1px solid rgba(255,255,255,0.05)`:"none",display:"flex",alignItems:"center",justifyContent:"center",transition:"width 0.6s cubic-bezier(.4,0,.2,1)"}}>
                <span style={{color:SC[k],fontSize:14,fontWeight:700,fontFamily:"monospace",textShadow:`0 0 12px ${SC[k]}50`}}>{a}</span>
              </div>
            ))}
          </div>
          <div style={{display:"flex",gap:16,marginTop:14,justifyContent:"center"}}>
            {SN.map((n,k)=>(
              <div key={k} style={{display:"flex",alignItems:"center",gap:6}}>
                <div style={{width:10,height:10,borderRadius:3,background:SC[k],boxShadow:`0 0 8px ${SC[k]}50`}}/>
                <span style={{color:C.dim,fontSize:12}}>{n}: <span style={{color:SC[k],fontWeight:600}}>{m.al[k]}</span></span>
              </div>
            ))}
          </div>
        </Glass>
        <Glass>
          <Label>Resource Flow</Label>
          <SankeyFlow allocation={m.al}/>
        </Glass>
      </div>

      {/* Per-Slice Charts */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(280px,1fr))",gap:14}}>
        {[
          {title:"eMBB Throughput",k:0,data:hist.map(h=>({v:h.tp[0]})),unit:"Mbps",target:100,color:C.embb},
          {title:"URLLC Latency",k:1,data:hist.map(h=>({v:h.lt[1]})),unit:"ms",target:1.0,color:C.urllc},
          {title:"mMTC Connected",k:2,data:hist.map(h=>({v:h.au[2]})),unit:"devices",target:200,color:C.mmtc},
        ].map(ch=>(
          <Glass key={ch.title}>
            <Label>{ch.title}</Label>
            <ResponsiveContainer width="100%" height={150}>
              <AreaChart data={ch.data.slice(-80)}>
                <defs><linearGradient id={`g${ch.k}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={ch.color} stopOpacity={.3}/><stop offset="100%" stopColor={ch.color} stopOpacity={0}/></linearGradient></defs>
                <CartesianGrid stroke={C.grid} strokeDasharray="3 3"/>
                <YAxis tick={{fill:C.dim,fontSize:10}} width={38}/>
                <Tooltip contentStyle={ttStyle} formatter={v=>[v.toFixed(2),ch.unit]}/>
                <ReferenceLine y={ch.target} stroke={C.bad} strokeDasharray="4 4" strokeWidth={1} opacity={0.6}/>
                <Area type="monotone" dataKey="v" stroke={ch.color} strokeWidth={2} fill={`url(#g${ch.k})`} dot={false}/>
              </AreaChart>
            </ResponsiveContainer>
          </Glass>
        ))}
      </div>

      {/* QoS Radar + Metrics Table */}
      <div style={{display:"grid",gridTemplateColumns:"1fr 2fr",gap:14}}>
        <Glass glow={C.accent}>
          <Label>QoS Satisfaction Radar</Label>
          <ResponsiveContainer width="100%" height={240}>
            <RadarChart data={radarData}>
              <PolarGrid stroke={C.grid} gridType="polygon"/>
              <PolarAngleAxis dataKey="axis" tick={{fill:C.dim,fontSize:10}}/>
              <PolarRadiusAxis tick={false} domain={[0,1]} axisLine={false}/>
              <Radar name="eMBB" dataKey="eMBB" stroke={C.embb} fill={C.embb} fillOpacity={0.15} strokeWidth={2}/>
              <Radar name="URLLC" dataKey="URLLC" stroke={C.urllc} fill={C.urllc} fillOpacity={0.15} strokeWidth={2}/>
              <Radar name="mMTC" dataKey="mMTC" stroke={C.mmtc} fill={C.mmtc} fillOpacity={0.15} strokeWidth={2}/>
            </RadarChart>
          </ResponsiveContainer>
        </Glass>

        <Glass>
          <Label>Live Slice Metrics</Label>
          <table style={{width:"100%",borderCollapse:"collapse"}}>
            <thead><tr>{["Slice","PRBs","Throughput","Latency","Drop Rate","Users","SLA"].map(h=>(
              <th key={h} style={{color:C.dim,fontSize:10,fontWeight:500,padding:"8px 10px",borderBottom:`1px solid ${C.border}`,textAlign:"left"}}>{h}</th>
            ))}</tr></thead>
            <tbody>{SN.map((n,k)=>{
              const sla=k===0?m.tp[0]>50:k===1?m.lt[1]<1:m.au[2]>150;
              return(
                <tr key={k} style={{background:k%2?"rgba(255,255,255,0.012)":"transparent"}}>
                  <td style={{padding:"10px"}}><Badge name={n} color={SC[k]}/></td>
                  <td style={{padding:"10px",fontFamily:"monospace",fontWeight:600,color:C.text}}>{m.al[k]}</td>
                  <td style={{padding:"10px",fontFamily:"monospace",color:C.text}}>{m.tp[k].toFixed(1)} <span style={{color:C.dim,fontSize:10}}>Mbps</span></td>
                  <td style={{padding:"10px",fontFamily:"monospace",color:k===1&&m.lt[1]>1?C.bad:C.text}}>{m.lt[k].toFixed(k===1?3:1)} <span style={{color:C.dim,fontSize:10}}>ms</span></td>
                  <td style={{padding:"10px",fontFamily:"monospace",color:C.text}}>{(m.dr[k]*100).toFixed(3)}%</td>
                  <td style={{padding:"10px",fontFamily:"monospace",color:C.text}}>{m.au[k]}/{[30,10,200][k]}</td>
                  <td style={{padding:"10px"}}><span style={{display:"flex",alignItems:"center",gap:3,color:sla?C.ok:C.bad,fontSize:12,fontWeight:600}}>
                    {sla?<CheckCircle size={13}/>:<AlertTriangle size={13}/>}{sla?"OK":"WARN"}</span></td>
                </tr>);})}</tbody>
          </table>
        </Glass>
      </div>
    </div>
  );
}

// ═══════════════════ TRAINING PAGE ═══════════════════
function TrainingPage() {
  const data = useMemo(()=>genTraining(),[]);
  const [vis,setVis] = useState(ALGOS.reduce((o,a)=>({...o,[a.n]:true}),{}));
  const [tab,setTab] = useState("reward");
  const smoothed = useMemo(()=>{
    const w=20; return data.map((d,i)=>{
      const o={episode:d.episode};
      ALGOS.forEach(a=>{const sl=data.slice(Math.max(0,i-w),i+1); o[a.n]=+(sl.reduce((s,p)=>s+p[a.n],0)/sl.length).toFixed(4);
        o[a.n+"_loss"]=+(sl.reduce((s,p)=>s+p[a.n+"_loss"],0)/sl.length).toFixed(4);
        o[a.n+"_eps"]=d[a.n+"_eps"];
      }); return o; });
  },[data]);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      {/* Tab selector */}
      <div style={{display:"flex",gap:6}}>
        {[{id:"reward",label:"Reward Convergence"},{id:"loss",label:"Loss Curves"},{id:"explore",label:"Exploration"}].map(t=>(
          <button key={t.id} onClick={()=>setTab(t.id)} style={{background:tab===t.id?`${C.accent}25`:"rgba(255,255,255,0.03)",border:`1px solid ${tab===t.id?C.accent:C.border}`,color:tab===t.id?C.accent:C.dim,borderRadius:10,padding:"7px 18px",cursor:"pointer",fontSize:12,fontWeight:tab===t.id?600:400}}>{t.label}</button>
        ))}
      </div>

      <Glass glow={C.accent}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
          <Label>{tab==="reward"?"Episode Reward (Smoothed MA-20)":tab==="loss"?"Training Loss":"\u03B5-Greedy Exploration Decay"}</Label>
          <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
            {ALGOS.map(a=>(
              <button key={a.n} onClick={()=>setVis(v=>({...v,[a.n]:!v[a.n]}))} style={{background:vis[a.n]?`${a.c}20`:"rgba(255,255,255,0.02)",border:`1px solid ${vis[a.n]?a.c:C.border}`,color:vis[a.n]?a.c:C.dim,borderRadius:7,padding:"3px 10px",cursor:"pointer",fontSize:10,fontWeight:600}}>{a.n}</button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={360}>
          <LineChart data={smoothed}>
            <CartesianGrid stroke={C.grid} strokeDasharray="3 3"/>
            <XAxis dataKey="episode" tick={{fill:C.dim,fontSize:10}} label={{value:"Episode",fill:C.dim,fontSize:11,position:"bottom"}}/>
            <YAxis tick={{fill:C.dim,fontSize:10}} domain={tab==="reward"?[0.2,0.9]:tab==="loss"?[0,0.5]:[0,1.1]}/>
            <Tooltip contentStyle={ttStyle}/>
            {ALGOS.map(a=>vis[a.n]&&(
              <Line key={a.n} dataKey={tab==="reward"?a.n:tab==="loss"?a.n+"_loss":a.n+"_eps"} stroke={a.c} strokeWidth={2} dot={false}/>
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Glass>

      {/* Algorithm cards */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:12}}>
        {ALGOS.map(a=>(
          <Glass key={a.n} glow={a.c} style={{padding:16}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
              <div style={{width:12,height:12,borderRadius:3,background:a.c,boxShadow:`0 0 8px ${a.c}50`}}/>
              <span style={{color:C.text,fontWeight:700,fontSize:15}}>{a.n}</span>
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
              <div><p style={{color:C.dim,fontSize:9}}>FINAL REWARD</p><p style={{color:a.c,fontWeight:700,fontSize:18,fontFamily:"monospace"}}>{a.fr}</p></div>
              <div><p style={{color:C.dim,fontSize:9}}>CONVERGE EP</p><p style={{color:C.text,fontWeight:600,fontSize:18,fontFamily:"monospace"}}>~{a.conv}</p></div>
            </div>
          </Glass>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════ COMPARISON PAGE ═══════════════════
function ComparisonPage() {
  const cdfData = useMemo(()=>genCDF(),[]);
  const bests = {rw:Math.max(...COMP.map(d=>d.rw)),et:Math.max(...COMP.map(d=>d.et)),ul:Math.min(...COMP.map(d=>d.ul)),mc:Math.max(...COMP.map(d=>d.mc)),sv:Math.min(...COMP.map(d=>d.sv))};

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      {/* Bar Charts */}
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(320px,1fr))",gap:14}}>
        {[{t:"eMBB Throughput (Mbps)",k:"et",c:C.embb},{t:"URLLC Latency (ms)",k:"ul",c:C.urllc},{t:"SLA Violation Rate (%)",k:"sv",c:C.bad}].map(ch=>(
          <Glass key={ch.t}>
            <Label>{ch.t}</Label>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={COMP} barSize={18}><CartesianGrid stroke={C.grid} strokeDasharray="3 3"/>
                <XAxis dataKey="n" tick={{fill:C.dim,fontSize:8}} angle={-25} textAnchor="end" height={50}/>
                <YAxis tick={{fill:C.dim,fontSize:10}}/><Tooltip contentStyle={ttStyle}/>
                <Bar dataKey={ch.k} radius={[4,4,0,0]}>{COMP.map((d,i)=><Cell key={i} fill={i<5?ch.c:`${ch.c}40`} fillOpacity={i<5?.85:.35}/>)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </Glass>
        ))}
      </div>

      {/* CDF Plot */}
      <Glass glow={C.urllc}>
        <Label>URLLC Latency CDF — Cumulative Distribution Function</Label>
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={cdfData}>
            <CartesianGrid stroke={C.grid} strokeDasharray="3 3"/>
            <XAxis dataKey="x" tick={{fill:C.dim,fontSize:10}} label={{value:"Latency (ms)",fill:C.dim,fontSize:11,position:"bottom"}}/>
            <YAxis tick={{fill:C.dim,fontSize:10}} domain={[0,1]} label={{value:"CDF",fill:C.dim,fontSize:11,angle:-90}}/>
            <Tooltip contentStyle={ttStyle}/>
            <ReferenceLine x={1.0} stroke={C.bad} strokeDasharray="4 4" label={{value:"1ms SLA",fill:C.bad,fontSize:10}}/>
            <Line dataKey="SAC" stroke={C.bad} strokeWidth={2} dot={false}/>
            <Line dataKey="DDQN" stroke={C.ok} strokeWidth={2} dot={false}/>
            <Line dataKey="Round Robin" stroke={C.dim} strokeWidth={2} dot={false} strokeDasharray="4 4"/>
            <Legend wrapperStyle={{fontSize:11,color:C.dim}}/>
          </LineChart>
        </ResponsiveContainer>
      </Glass>

      {/* Comparison Table */}
      <Glass glow={C.accent}>
        <Label>Algorithm Comparison Matrix</Label>
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",minWidth:750}}>
            <thead><tr>{["Algorithm","Avg Reward","eMBB Tput","URLLC Lat","mMTC Conn","SLA Viol%","Converge"].map(h=>(
              <th key={h} style={{color:C.dim,fontSize:10,fontWeight:600,padding:"10px 12px",borderBottom:`1px solid ${C.border}`,textAlign:"left",letterSpacing:.5}}>{h}</th>
            ))}</tr></thead>
            <tbody>{COMP.map((d,i)=>{
              const hi=(k,v,best,lower)=>v===best?{color:lower?C.urllc:C.ok,fontWeight:700,textShadow:`0 0 8px ${lower?C.urllc:C.ok}40`}:{};
              return(
                <tr key={i} style={{background:i%2?"rgba(255,255,255,0.012)":"transparent",opacity:i>=5?.7:1}}>
                  <td style={{padding:"10px 12px",fontWeight:600,color:i<5?C.text:C.dim}}>{d.n}</td>
                  <td style={{padding:"10px 12px",fontFamily:"monospace",...hi("rw",d.rw,bests.rw)}}>{d.rw}</td>
                  <td style={{padding:"10px 12px",fontFamily:"monospace",...hi("et",d.et,bests.et)}}>{d.et}</td>
                  <td style={{padding:"10px 12px",fontFamily:"monospace",...hi("ul",d.ul,bests.ul,true)}}>{d.ul}</td>
                  <td style={{padding:"10px 12px",fontFamily:"monospace",...hi("mc",d.mc,bests.mc)}}>{d.mc}</td>
                  <td style={{padding:"10px 12px",fontFamily:"monospace",color:d.sv<=1?C.ok:d.sv>5?C.bad:C.warn}}>{d.sv}%</td>
                  <td style={{padding:"10px 12px",fontFamily:"monospace",color:C.dim}}>{d.cv??"N/A"}</td>
                </tr>);})}</tbody>
          </table>
        </div>
      </Glass>
    </div>
  );
}

// ═══════════════════ NETWORK PAGE ═══════════════════
function NetworkPage({m}) {
  const [ues] = useState(()=>{const u=[];
    for(let i=0;i<30;i++)u.push({x:(Math.random()-.5)*280,y:(Math.random()-.5)*280,t:0});
    for(let i=0;i<10;i++)u.push({x:(Math.random()-.5)*200,y:(Math.random()-.5)*200,t:1});
    for(let i=0;i<40;i++)u.push({x:(Math.random()-.5)*320,y:(Math.random()-.5)*320,t:2});
    return u;});

  const grid = useMemo(()=>{const r=[];for(let t=0;t<14;t++){const row=[];let idx=0;
    for(let k=0;k<3;k++){for(let p=0;p<m.al[k];p++){row.push({s:k,a:Math.random()<[.8,.9,.4][k]});idx++;}}
    while(row.length<100)row.push({s:-1,a:false});r.push(row);}return r;},[m.al,m.step]);

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1.5fr",gap:14}}>
        {/* Cell Topology */}
        <Glass glow={C.accent}>
          <Label>Cell Topology — 500m Radius</Label>
          <div style={{position:"relative",width:"100%",paddingBottom:"100%",maxWidth:380,margin:"0 auto"}}>
            <svg viewBox="-180 -180 360 360" style={{position:"absolute",top:0,left:0,width:"100%",height:"100%"}}>
              <defs>
                <radialGradient id="cellGrad"><stop offset="0%" stopColor={C.accent} stopOpacity=".06"/><stop offset="100%" stopColor={C.accent} stopOpacity="0"/></radialGradient>
              </defs>
              <circle cx="0" cy="0" r="165" fill="url(#cellGrad)"/>
              <circle cx="0" cy="0" r="165" fill="none" stroke={C.border} strokeWidth="1" strokeDasharray="4 4"/>
              <circle cx="0" cy="0" r="110" fill="none" stroke={C.border} strokeWidth=".5" strokeDasharray="2 4" opacity=".4"/>
              <circle cx="0" cy="0" r="55" fill="none" stroke={C.border} strokeWidth=".5" strokeDasharray="2 4" opacity=".25"/>
              {/* Tower */}
              <polygon points="0,-14 9,7 -9,7" fill={C.accent} opacity=".9"/>
              <circle cx="0" cy="0" r="5" fill={C.accent}><animate attributeName="r" values="5;10;5" dur="2s" repeatCount="indefinite"/><animate attributeName="opacity" values="1;.2;1" dur="2s" repeatCount="indefinite"/></circle>
              {ues.map((u,i)=><circle key={i} cx={u.x*.5} cy={u.y*.5} r={u.t===2?1.8:3.2} fill={SC[u.t]} opacity={.65+Math.random()*.35}/>)}
            </svg>
          </div>
          <div style={{display:"flex",justifyContent:"center",gap:14,marginTop:8}}>
            {SN.map((n,k)=><div key={k} style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:C.dim}}>
              <div style={{width:7,height:7,borderRadius:"50%",background:SC[k]}}/>{n} ({[30,10,40][k]})
            </div>)}
          </div>
        </Glass>

        {/* Resource Grid */}
        <Glass>
          <Label>OFDMA Resource Grid — PRB × Time Slot</Label>
          <div style={{display:"flex",flexDirection:"column",gap:1}}>
            {grid.map((row,t)=>(
              <div key={t} style={{display:"flex",gap:.5,height:16}}>
                {row.map((cell,p)=>(
                  <div key={p} style={{flex:1,minWidth:0,background:cell.s>=0?`${SC[cell.s]}${cell.a?"55":"18"}`:C.grid,borderRadius:.5,transition:"background 0.4s"}}/>
                ))}
              </div>
            ))}
          </div>
          <div style={{display:"flex",justifyContent:"space-between",marginTop:6,padding:"0 2px"}}>
            <span style={{color:C.dim,fontSize:9}}>PRB 0</span>
            <span style={{color:C.dim,fontSize:9}}>← 100 Physical Resource Blocks →</span>
            <span style={{color:C.dim,fontSize:9}}>PRB 99</span>
          </div>
        </Glass>
      </div>

      {/* Reward History */}
      <Glass>
        <Label>Agent Reward History</Label>
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={Array.from({length:80},(_,i)=>({step:m.step-80+i,reward:m.rw+Math.sin(i/8)*.02+(Math.random()-.5)*.03}))}>
            <defs><linearGradient id="rwG" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={C.accent} stopOpacity={.3}/><stop offset="100%" stopColor={C.accent} stopOpacity={0}/></linearGradient></defs>
            <CartesianGrid stroke={C.grid} strokeDasharray="3 3"/>
            <YAxis tick={{fill:C.dim,fontSize:10}} domain={["auto","auto"]} width={40}/>
            <Tooltip contentStyle={ttStyle}/>
            <Area type="monotone" dataKey="reward" stroke={C.accent} strokeWidth={2} fill="url(#rwG)" dot={false}/>
          </AreaChart>
        </ResponsiveContainer>
      </Glass>
    </div>
  );
}

// ═══════════════════ SCENARIOS PAGE ═══════════════════
function ScenariosPage({onScenario,m}) {
  const scenes = [
    {id:"flash_crowd",n:"Flash Crowd",d:"5× traffic surge on eMBB — simulates stadium/concert event",icon:Zap,c:C.embb},
    {id:"urllc_emergency",n:"URLLC Emergency",d:"Critical burst of URLLC packets — factory emergency scenario",icon:AlertTriangle,c:C.urllc},
    {id:"mmtc_storm",n:"mMTC Storm",d:"All 200 IoT devices reporting simultaneously",icon:Wifi,c:C.mmtc},
    {id:"network_degradation",n:"Network Degradation",d:"Reduce capacity from 100→50 PRBs — partial cell failure",icon:Activity,c:C.bad},
    {id:"reset",n:"Reset Normal",d:"Restore default network conditions",icon:RotateCcw,c:C.ok},
  ];

  return (
    <div style={{display:"flex",flexDirection:"column",gap:16}}>
      <Glass>
        <Label>Scenario Testing — Stress Test DRL Adaptability</Label>
        <p style={{color:C.dim,fontSize:13,marginBottom:18}}>Trigger real-time network events to observe how the DRL agent dynamically re-allocates resources to maintain QoS.</p>
        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(220px,1fr))",gap:12}}>
          {scenes.map(s=>(
            <button key={s.id} onClick={()=>onScenario(s.id)}
              style={{background:`${s.c}08`,border:`1px solid ${s.c}25`,borderRadius:14,padding:18,cursor:"pointer",textAlign:"left",transition:"all 0.3s"}}
              onMouseEnter={e=>{e.currentTarget.style.borderColor=s.c;e.currentTarget.style.transform="translateY(-2px)";e.currentTarget.style.boxShadow=`0 8px 30px ${s.c}15`;}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor=`${s.c}25`;e.currentTarget.style.transform="none";e.currentTarget.style.boxShadow="none";}}>
              <s.icon size={22} color={s.c} style={{marginBottom:10}}/>
              <p style={{color:C.text,fontSize:14,fontWeight:600,marginBottom:5}}>{s.n}</p>
              <p style={{color:C.dim,fontSize:11,lineHeight:1.5}}>{s.d}</p>
            </button>
          ))}
        </div>
      </Glass>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <Glass>
          <Label>Current Network State</Label>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
            {[{l:"eMBB Throughput",v:`${m.tp[0].toFixed(1)} Mbps`,c:C.embb},
              {l:"URLLC Latency",v:`${m.lt[1].toFixed(3)} ms`,c:C.urllc},
              {l:"mMTC Devices",v:`${m.au[2]}/200`,c:C.mmtc},
              {l:"Utilization",v:`${(m.ut*100).toFixed(1)}%`,c:C.accent},
              {l:"Agent Reward",v:m.rw.toFixed(3),c:C.ok},
              {l:"Epsilon (ε)",v:m.eps.toFixed(3),c:C.purple},
            ].map(i=>(
              <div key={i.l} style={{background:`${i.c}08`,borderRadius:10,padding:"12px 14px",border:`1px solid ${i.c}18`}}>
                <p style={{color:C.dim,fontSize:10,marginBottom:4}}>{i.l}</p>
                <p style={{color:i.c,fontSize:18,fontWeight:700,fontFamily:"monospace"}}>{i.v}</p>
              </div>
            ))}
          </div>
        </Glass>

        <Glass>
          <Label>Agent Decision Distribution</Label>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={[
              {action:"↓ eMBB",count:Math.round(20+Math.random()*15)},
              {action:"= eMBB",count:Math.round(40+Math.random()*20)},
              {action:"↑ eMBB",count:Math.round(25+Math.random()*15)},
              {action:"↓ URLLC",count:Math.round(10+Math.random()*10)},
              {action:"= URLLC",count:Math.round(35+Math.random()*15)},
              {action:"↑ URLLC",count:Math.round(40+Math.random()*20)},
              {action:"↓ mMTC",count:Math.round(15+Math.random()*10)},
              {action:"= mMTC",count:Math.round(45+Math.random()*15)},
              {action:"↑ mMTC",count:Math.round(20+Math.random()*15)},
            ]} barSize={14}>
              <CartesianGrid stroke={C.grid} strokeDasharray="3 3"/>
              <XAxis dataKey="action" tick={{fill:C.dim,fontSize:8}} angle={-30} textAnchor="end" height={45}/>
              <YAxis tick={{fill:C.dim,fontSize:10}}/>
              <Tooltip contentStyle={ttStyle}/>
              <Bar dataKey="count" radius={[3,3,0,0]}>
                {[C.embb,C.embb,C.embb,C.urllc,C.urllc,C.urllc,C.mmtc,C.mmtc,C.mmtc].map((c,i)=><Cell key={i} fill={c} opacity={.7}/>)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Glass>
      </div>
    </div>
  );
}

// ═══════════════════ MAIN APP ═══════════════════
const NAV = [
  {id:"dashboard",label:"Dashboard",icon:Gauge},
  {id:"training",label:"Training",icon:Brain},
  {id:"comparison",label:"Comparison",icon:BarChart3},
  {id:"network",label:"Network",icon:Network},
  {id:"scenarios",label:"Scenarios",icon:Zap},
];

export default function App() {
  const [page,setPage] = useState("dashboard");
  const [run,setRun] = useState(false);
  const [spd,setSpd] = useState(1);
  const [sb,setSb] = useState(true);
  const sim = useRef(null);
  const iv = useRef(null);
  const [m,setM] = useState({step:0,al:[33,33,34],tp:[80,8,12],lt:[35,1.2,400],dr:[.015,.0005,.06],au:[25,10,160],rw:.3,ut:.67,sla:true,mmpp:0,tt:100,eps:1,loss:.5});
  const [hist,setHist] = useState([]);

  useEffect(()=>{sim.current=createSim();return()=>{if(iv.current)clearInterval(iv.current);};},[]);

  const start=useCallback(()=>{
    setRun(true);
    if(iv.current)clearInterval(iv.current);
    iv.current=setInterval(()=>{if(sim.current){const d=sim.current.tick();setM(d);setHist(p=>{const n=[...p,d];return n.length>300?n.slice(-300):n;});}},Math.max(16,100/spd));
  },[spd]);

  const stop=useCallback(()=>{setRun(false);if(iv.current)clearInterval(iv.current);},[]);
  const reset=useCallback(()=>{stop();sim.current?.reset();setM({step:0,al:[33,33,34],tp:[80,8,12],lt:[35,1.2,400],dr:[.015,.0005,.06],au:[25,10,160],rw:.3,ut:.67,sla:true,mmpp:0,tt:100,eps:1,loss:.5});setHist([]);},[stop]);

  useEffect(()=>{if(run){if(iv.current)clearInterval(iv.current);
    iv.current=setInterval(()=>{if(sim.current){const d=sim.current.tick();setM(d);setHist(p=>{const n=[...p,d];return n.length>300?n.slice(-300):n;});}},Math.max(16,100/spd));}},[spd,run]);

  const onScenario=useCallback((id)=>{sim.current?.scenario(id);},[]);

  return (
    <div style={{display:"flex",height:"100vh",background:C.bg,color:C.text,fontFamily:"'Segoe UI',-apple-system,BlinkMacSystemFont,sans-serif",overflow:"hidden"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
        @keyframes glowText{0%,100%{text-shadow:0 0 20px rgba(124,58,237,.5)}50%{text-shadow:0 0 40px rgba(124,58,237,.9),0 0 80px rgba(124,58,237,.3)}}
        ::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:rgba(255,255,255,.08);border-radius:3px}
        *{box-sizing:border-box;margin:0;padding:0}`}</style>

      {/* Sidebar */}
      <div style={{width:sb?210:60,minWidth:sb?210:60,background:"rgba(6,10,20,0.97)",borderRight:`1px solid ${C.border}`,display:"flex",flexDirection:"column",transition:"all 0.3s",overflow:"hidden"}}>
        <div style={{padding:"18px 14px",borderBottom:`1px solid ${C.border}`,display:"flex",alignItems:"center",gap:10,cursor:"pointer",whiteSpace:"nowrap"}} onClick={()=>setSb(v=>!v)}>
          <Radio size={20} color={C.accent}/>
          {sb&&<span style={{fontWeight:800,fontSize:17,background:`linear-gradient(135deg,${C.accent},${C.embb})`,WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",animation:"glowText 3s ease-in-out infinite"}}>IntelliSlice</span>}
        </div>
        <div style={{flex:1,padding:"10px 6px",display:"flex",flexDirection:"column",gap:3}}>
          {NAV.map(item=>{const act=page===item.id;return(
            <button key={item.id} onClick={()=>setPage(item.id)} style={{display:"flex",alignItems:"center",gap:11,padding:"9px 12px",borderRadius:10,background:act?`${C.accent}18`:"transparent",border:"none",color:act?C.accent:C.dim,cursor:"pointer",transition:"all 0.2s",whiteSpace:"nowrap",fontSize:13,fontWeight:act?600:400}}>
              <item.icon size={17}/>{sb&&item.label}
            </button>);})}
        </div>
        {sb&&<div style={{padding:14,borderTop:`1px solid ${C.border}`}}>
          <div style={{background:`${C.ok}0D`,border:`1px solid ${C.ok}25`,borderRadius:10,padding:"10px 12px"}}>
            <p style={{color:C.dim,fontSize:9,textTransform:"uppercase",letterSpacing:1.5}}>Active Agent</p>
            <p style={{color:C.ok,fontWeight:700,fontSize:14}}>DDQN + PER</p>
            <p style={{color:C.dim,fontSize:10,marginTop:3}}>ε = {m.eps}</p>
          </div>
        </div>}
      </div>

      {/* Main */}
      <div style={{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"}}>
        <div style={{padding:"10px 22px",borderBottom:`1px solid ${C.border}`,display:"flex",alignItems:"center",gap:14,background:"rgba(6,10,20,.5)",backdropFilter:"blur(10px)"}}>
          <h1 style={{fontSize:16,fontWeight:600}}>{NAV.find(n=>n.id===page)?.label}</h1>
          <div style={{flex:1}}/>
          <div style={{display:"flex",alignItems:"center",gap:16,fontSize:12,fontFamily:"'JetBrains Mono',monospace"}}>
            <span style={{color:C.dim}}>Reward: <span style={{color:m.rw>.6?C.ok:C.warn,fontWeight:600}}>{m.rw.toFixed(3)}</span></span>
            <span style={{color:C.dim}}>Loss: <span style={{color:C.purple}}>{m.loss}</span></span>
          </div>
        </div>
        <div style={{flex:1,overflow:"auto",padding:"18px 22px"}}>
          {page==="dashboard"&&<DashboardPage m={m} hist={hist} run={run} onStart={start} onStop={stop} onReset={reset} spd={spd} setSpd={setSpd}/>}
          {page==="training"&&<TrainingPage/>}
          {page==="comparison"&&<ComparisonPage/>}
          {page==="network"&&<NetworkPage m={m}/>}
          {page==="scenarios"&&<ScenariosPage onScenario={onScenario} m={m}/>}
        </div>
      </div>
    </div>
  );
}