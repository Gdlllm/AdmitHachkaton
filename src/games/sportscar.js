/** A rear-engined sports coupé in the spirit of the classic 911 silhouette,
 * built from extruded side profiles (no external model): sloping fastback,
 * round headlights, a full-width rear light bar and a soft underglow. Front is +Z, metres, origin on the ground. */
import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

const LENGTH = 4.5, WIDTH = 1.84, CABIN = 1.32;

function extrude(points, width, bevel) {
  const shape = new THREE.Shape();
  shape.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const p = points[i];
    if (p.length === 4) shape.quadraticCurveTo(p[0], p[1], p[2], p[3]); else shape.lineTo(p[0], p[1]);
  }
  shape.closePath();
  const geo = new THREE.ExtrudeGeometry(shape, { depth: width - bevel * 2, bevelEnabled: true, bevelThickness: bevel, bevelSize: bevel, bevelSegments: 5, curveSegments: 18 });
  // Shape X = length (front +), Y = height, extruded along Z = width → length along Z, width along X.
  geo.rotateY(-Math.PI / 2);
  geo.translate((width - bevel * 2) / 2, 0, 0);
  geo.computeVertexNormals();
  return geo;
}

/** renderer: used once to prefilter the studio reflections that make the paint shine at night. */
export function createSportsCar(renderer, { color = '#ff2238', glow = '#ff2bd6' } = {}) {
  const car = new THREE.Group();
  const pmrem = new THREE.PMREMGenerator(renderer), env = pmrem.fromScene(new RoomEnvironment(), .04).texture; pmrem.dispose();
  const paint = new THREE.MeshPhysicalMaterial({ color, metalness: .5, roughness: .2, clearcoat: 1, clearcoatRoughness: .06, envMap: env, envMapIntensity: .9,
    emissive: '#3a0008', emissiveIntensity: .5 });
  const glass = new THREE.MeshPhysicalMaterial({ color: '#0b0e16', metalness: .2, roughness: .05, clearcoat: 1, envMap: env, envMapIntensity: .8 });
  const dark = new THREE.MeshStandardMaterial({ color: '#15161a', roughness: .8 });
  const rim = new THREE.MeshStandardMaterial({ color: '#c9ccd3', metalness: .9, roughness: .25 });
  // Lower body: bumper, long rising front wings, wide hips over the rear wheels, ducktail.
  const body = extrude([[2.25, .28], [2.3, .52, 2.2, .62], [1.1, .8, .2, .86], [-1.5, .9, -2.15, .82], [-2.3, .72], [-2.28, .3]], WIDTH, .13);
  car.add(new THREE.Mesh(body, paint));
  // Cabin: windscreen, rounded roof, the long fastback to the tail.
  const cabin = extrude([[.9, .84], [.35, 1.12, -.1, 1.26], [-.5, 1.3, -1.0, 1.18], [-1.7, .96, -2.0, .86], [.9, .84]], CABIN, .1);
  car.add(new THREE.Mesh(cabin, glass));
  const roof = extrude([[.25, 1.2], [-.1, 1.29, -.5, 1.31], [-.95, 1.24, -1.1, 1.16], [.25, 1.2]], CABIN - .12, .05);
  car.add(new THREE.Mesh(roof, paint));
  // Wheels with bright rims.
  const tyre = new THREE.CylinderGeometry(.34, .34, .26, 24); tyre.rotateZ(Math.PI / 2);
  const hub = new THREE.CylinderGeometry(.22, .22, .27, 18); hub.rotateZ(Math.PI / 2);
  for (const [x, z] of [[-.84, 1.38], [.84, 1.38], [-.86, -1.42], [.86, -1.42]]) {
    const w = new THREE.Mesh(tyre, dark); w.position.set(x, .34, z); car.add(w);
    const h = new THREE.Mesh(hub, rim); h.position.set(x * 1.01, .34, z); car.add(h);
  }
  // Round headlights, rear light bar, underglow.
  const lamp = new THREE.MeshStandardMaterial({ color: '#ffffff', emissive: '#e8f4ff', emissiveIntensity: 3 });
  for (const x of [-.6, .6]) {
    const l = new THREE.Mesh(new THREE.SphereGeometry(.14, 20, 12), lamp); l.scale.set(1, 1, .45); l.position.set(x, .64, 2.4); car.add(l);
  }
  const tail = new THREE.MeshStandardMaterial({ color: '#ff1e3c', emissive: '#ff1e3c', emissiveIntensity: 3 });
  const bar = new THREE.Mesh(new THREE.BoxGeometry(1.7, .09, .06), tail); bar.position.set(0, .66, -2.46); car.add(bar);
  for (const x of [-.72, .72]) { const t = new THREE.Mesh(new THREE.BoxGeometry(.28, .14, .06), tail); t.position.set(x, .66, -2.45); car.add(t); }
  car.userData.tail = tail;
  // Soft round underglow: a radial gradient, brightest under the middle of the car.
  const halo = document.createElement('canvas'); halo.width = halo.height = 128;
  const hg = halo.getContext('2d'), grad = hg.createRadialGradient(64, 64, 4, 64, 64, 64);
  grad.addColorStop(0, 'rgba(255,255,255,1)'); grad.addColorStop(.5, 'rgba(255,255,255,.45)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
  hg.fillStyle = grad; hg.fillRect(0, 0, 128, 128);
  const underglow = new THREE.Mesh(new THREE.PlaneGeometry(WIDTH + 1.6, LENGTH + 1.2),
    new THREE.MeshBasicMaterial({ color: glow, map: new THREE.CanvasTexture(halo), transparent: true, opacity: .8, blending: THREE.AdditiveBlending, depthWrite: false }));
  underglow.rotation.x = -Math.PI / 2; underglow.position.y = .03; car.add(underglow);
  car.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = false; } });
  underglow.castShadow = false;
  car.userData.glow = { plane: underglow };
  return car;
}
