export const CONNECTIONS = [[0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],[11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],[11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],[27,29],[28,30],[29,31],[30,32],[27,31],[28,32]];
export const CORE = [11,12,13,14,15,16,23,24,25,26,27,28];
export const visible = p => Boolean(p && Number.isFinite(p.x) && Number.isFinite(p.y) && (p.visibility ?? 0) >= 0.55 && (p.presence ?? 1) >= 0.5 && p.x > 0.015 && p.x < 0.985 && p.y > 0.015 && p.y < 0.985);
const clamp = (v, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));
const distance = (a,b) => Math.hypot(a.x-b.x,a.y-b.y);
export function angle(a,b,c) {
  const ab = {x:a.x-b.x,y:a.y-b.y}, cb = {x:c.x-b.x,y:c.y-b.y};
  const norm = Math.hypot(ab.x,ab.y)*Math.hypot(cb.x,cb.y);
  return norm < 1e-8 ? 0 : Math.acos(clamp((ab.x*cb.x+ab.y*cb.y)/norm,-1,1))*180/Math.PI;
}

// Adaptive low-pass: fast movement gets less smoothing than a stationary joint.
export class PoseSmoother {
  reset() { this.previous = null; this.time = null; }
  update(points, time) {
    if (!this.previous || time-this.time > 300 || points.length !== this.previous.length) {
      this.previous = points.map(p=>({...p})); this.time = time; return points;
    }
    const dt = Math.max((time-this.time)/1000,0.001);
    const result = points.map((p,i)=> {
      if (!visible(p) || !visible(this.previous[i])) return {...p};
      const old = this.previous[i];
      const speed = Math.hypot(p.x-old.x,p.y-old.y)/dt;
      const cutoff = 2.2 + speed*9;
      const alpha = 1 / (1 + 1/(2*Math.PI*cutoff*dt));
      return {...p, x:old.x+alpha*(p.x-old.x), y:old.y+alpha*(p.y-old.y), z:old.z+alpha*((p.z ?? 0)-old.z)};
    });
    this.previous=result; this.time=time; return result;
  }
}

export class MotionRecognizer {
  constructor() { this.reset(); }
  reset() {
    this.states = Object.fromEntries(['reach','spread','squat'].map(id=>[id,{ since:null, released:null, latched:false }]));
    this.standingSince = null; this.squatReady = false; this.lastTime = null;
  }
  update(points,time,aspect=16/9) {
    if (this.lastTime !== null && time-this.lastTime > 500) this.reset();
    this.lastTime=time;
    const valid = ids=>ids.every(i=>visible(points[i]));
    const p = points.map(v=>({...v,x:v.x*aspect}));
    const torsoOK = valid([11,12,23,24]);
    const armsOK = torsoOK && valid([13,14,15,16]);
    const legsOK = torsoOK && valid([25,26,27,28]);
    const torso = torsoOK ? Math.max(0.08,((p[23].y+p[24].y)-(p[11].y+p[12].y))/2) : 1;
    let reach=0, spread=0, squat=0, knees=null;
    let reachHint='Покажи плечи, таз и обе кисти', spreadHint=reachHint, squatHint='Отойди: нужны таз, колени и стопы';
    let reachOn=false, spreadOn=false, squatOn=false;
    if (armsOK) {
      const height = Math.min((p[11].y-p[15].y)/torso,(p[12].y-p[16].y)/torso);
      reach=clamp((height+0.15)/0.8); reachOn=height>0.55;
      reachHint=reachOn ? 'Отлично. Удержи обе руки наверху' : height>0 ? 'Подними обе кисти выше головы' : 'Подними обе руки над головой';
      const off = Math.max(Math.abs(p[15].y-p[11].y),Math.abs(p[16].y-p[12].y))/torso;
      const elbow = Math.min(angle(p[11],p[13],p[15]),angle(p[12],p[14],p[16]));
      const length = Math.min(distance(p[11],p[15]),distance(p[12],p[16]))/torso;
      const outward = (p[15].x-p[11].x)*(p[11].x-p[12].x)>0 && (p[16].x-p[12].x)*(p[12].x-p[11].x)>0;
      spread=outward ? clamp(1-off)*clamp((elbow-75)/80)*clamp(length/0.8) : 0;
      spreadOn=outward && off<0.23 && elbow>150 && length>0.7;
      spreadHint=!outward ? 'Разведи руки в разные стороны' : elbow<150 ? 'Выпрями оба локтя' : off>=0.23 ? 'Держи кисти на высоте плеч' : 'Отлично. Удержи руки в стороны';
    }
    if (legsOK) {
      knees=(angle(p[23],p[25],p[27])+angle(p[24],p[26],p[28]))/2;
      if (knees>155) {
        this.standingSince ??= time;
        if (time-this.standingSince>200) this.squatReady=true;
      } else this.standingSince=null;
      squat=clamp((168-knees)/55);
      squatOn=this.squatReady && knees<118;
      squatHint=!this.squatReady ? 'Сначала встань прямо, затем присядь' : knees>145 ? 'Согни колени и опусти таз' : knees>=118 ? 'Опусти таз немного ниже' : 'Присед распознан. Теперь выпрямись';
    } else { this.standingSince=null; this.squatReady=false; }
    const specs=[['reach',armsOK,reachOn,reach,reachHint],['spread',armsOK,spreadOn,spread,spreadHint],['squat',legsOK,squatOn,squat,squatHint]];
    const events=[];
    const gestures=specs.map(([id,valid,on,progress,hint])=> {
      const state=this.states[id];
      if (!valid) { state.since=null; state.released=null; }
      else if (on) {
        state.released=null; state.since ??=time;
        if (!state.latched && time-state.since>=280) {
          state.latched=true; events.push(id);
        }
      } else {
        state.since=null; state.released ??=time;
        const neutral = id !== 'squat' || knees>155;
        if (neutral && time-state.released>=220) state.latched=false;
      }
      return {id,valid,active:valid && on && state.latched,progress, hint,hold:state.since===null?0:clamp((time-state.since)/280)};
    });
    const seen=CORE.filter(i=>visible(points[i])).length;
    let hint = points.length ? 'Тело найдено. Попробуй одно из движений справа' : 'Встань перед камерой так, чтобы тебя было видно';
    if (points.length && !torsoOK) hint='Отойди от камеры, чтобы были видны плечи и таз';
    else if (points.length && !legsOK) hint='Верх тела виден. Отойди, чтобы в кадр попали ноги';
    else if (points.length && !armsOK) hint='Держи обе кисти в кадре, не закрывай их телом';
    return {gestures,events,seen,quality:Math.round(seen/CORE.length*100),torsoOK,armsOK,legsOK,hint,knees};
  }
}
