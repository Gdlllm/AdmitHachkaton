/** A table tennis bat: round-ish blade with red rubber on the front (+Z) and
 * black on the back, a wooden edge and a flared handle. The handle end is at
 * the origin, the blade points up +Y. `ghost` makes it see-through with a
 * bright rim, for the player's own bat right in front of the camera. */
import * as THREE from 'three';

export function createPaddle({ ghost = false } = {}) {
  const bat = new THREE.Group();
  const see = ghost ? { transparent: true, opacity: .34, depthWrite: false } : {};
  const red = new THREE.MeshStandardMaterial({ color: '#d0202a', roughness: .75, ...see });
  const black = new THREE.MeshStandardMaterial({ color: '#141416', roughness: .8, ...see });
  const wood = new THREE.MeshStandardMaterial({ color: '#c89a62', roughness: .6, ...(ghost ? { transparent: true, opacity: .8 } : {}) });
  const blade = new THREE.Group(); blade.position.y = .175; bat.add(blade);
  const face = (mat, z) => { const m = new THREE.Mesh(new THREE.CircleGeometry(.078, 40), mat); m.scale.set(.96, 1, 1); m.position.z = z; if (z < 0) m.rotation.y = Math.PI; blade.add(m); };
  face(red, .0045); face(black, -.0045);
  const edge = new THREE.Mesh(new THREE.TorusGeometry(.078, .0045, 6, 48), wood); edge.scale.set(.96, 1, 1); blade.add(edge);
  if (ghost) {
    const rim = new THREE.Mesh(new THREE.TorusGeometry(.08, .003, 6, 48), new THREE.MeshBasicMaterial({ color: '#7cf2d8' }));
    rim.scale.set(.96, 1, 1); blade.add(rim);
    const dot = new THREE.Mesh(new THREE.RingGeometry(.008, .012, 20), new THREE.MeshBasicMaterial({ color: '#7cf2d8', transparent: true, opacity: .8, side: THREE.DoubleSide }));
    dot.position.z = .006; blade.add(dot);
  }
  const handle = new THREE.Mesh(new THREE.BoxGeometry(.028, .1, .024), wood); handle.position.y = .05; bat.add(handle);
  const flare = new THREE.Mesh(new THREE.BoxGeometry(.036, .03, .026), wood); flare.position.y = .095; bat.add(flare);
  bat.traverse(o => { if (o.isMesh) o.castShadow = !ghost; });
  bat.userData.blade = blade;
  return bat;
}
