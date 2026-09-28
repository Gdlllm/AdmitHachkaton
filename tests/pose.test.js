import test from 'node:test';
import assert from 'node:assert/strict';
import { MotionRecognizer, PoseSmoother, angle } from '../src/tracking/pose.js';
function pose(kind='neutral') {
  const p=Array.from({length:33},()=>({x:.5,y:.15,z:0,visibility:1,presence:1}));
  const set=(i,x,y)=>Object.assign(p[i],{x,y});
  set(11,.4,.3);set(12,.6,.3);set(23,.43,.55);set(24,.57,.55);
  set(13,.39,.42);set(14,.61,.42);set(15,.38,.53);set(16,.62,.53);
  set(25,.43,.72);set(26,.57,.72);set(27,.43,.9);set(28,.57,.9);
  if(kind==='reach'){set(13,.4,.2);set(14,.6,.2);set(15,.4,.08);set(16,.6,.08);}
  if(kind==='spread'){set(13,.28,.3);set(14,.72,.3);set(15,.15,.3);set(16,.85,.3);}
  if(kind==='squat'){set(23,.43,.66);set(24,.57,.66);set(25,.31,.72);set(26,.69,.72);}
  return p;
}
test('time-based hold fires once, neutral re-arms',()=>{
  const r=new MotionRecognizer();
  assert.deepEqual(r.update(pose('reach'),0).events,[]);
  assert.deepEqual(r.update(pose('reach'),150).events,[]);
  assert.deepEqual(r.update(pose('reach'),300).events,['reach']);
  assert.deepEqual(r.update(pose('reach'),600).events,[]);
  r.update(pose(),700);r.update(pose(),950);r.update(pose('reach'),1000);
  assert.deepEqual(r.update(pose('reach'),1300).events,['reach']);
});
test('missing or out-of-frame wrist cannot produce an event',()=>{
  for(const mutation of [{visibility:.1},{x:1.2},{presence:.1}]){
    const r=new MotionRecognizer(),p=pose('reach');Object.assign(p[15],mutation);
    r.update(p,0);const result=r.update(p,400);
    assert.deepEqual(result.events,[]);assert.equal(result.armsOK,false);
  }
});
test('tracking loss interrupts hold and never reuses old landmarks',()=>{
  const r=new MotionRecognizer();r.update(pose('spread'),0);
  assert.deepEqual(r.update([],200).events,[]);
  assert.deepEqual(r.update(pose('spread'),350).events,[]);
  assert.deepEqual(r.update(pose('spread'),650).events,['spread']);
});
test('squat requires standing baseline and return to standing for next repetition',()=>{
  const r=new MotionRecognizer();r.update(pose('squat'),0);
  assert.deepEqual(r.update(pose('squat'),400).events,[]);
  r.update(pose(),450);r.update(pose(),700);r.update(pose('squat'),800);
  assert.deepEqual(r.update(pose('squat'),1100).events,['squat']);
  assert.deepEqual(r.update(pose('squat'),1300).events,[]);
  r.update(pose(),1400);r.update(pose(),1700);r.update(pose('squat'),1800);
  assert.deepEqual(r.update(pose('squat'),2100).events,['squat']);
});
test('2D joint angles account for input aspect ratio',()=>{
  assert.equal(angle({x:0,y:0},{x:1,y:0},{x:1,y:1}),90);
  const p=pose('squat');const p2=p.map(v=>({...v,x:v.x/2}));
  const a=new MotionRecognizer().update(p,0,1);
  const b=new MotionRecognizer().update(p2,0,2);
  assert.ok(Math.abs(a.knees-b.knees)<1e-8);
});
test('smoothing never retains high visibility for a newly hidden point',()=>{
  const s=new PoseSmoother();s.update(pose(),0);const next=pose();next[15].visibility=.1;
  assert.equal(s.update(next,33)[15].visibility,.1);
  assert.deepEqual(s.update([],66),[]);
});
