// Depth-tested wireframe of the actual decoded surface, not a generated body.
// Drawn in source-image coordinates at the pixel size of the on-screen video
// rectangle; the 2D renderer then applies the same contain/mirror transform as
// for the camera image.
export class MeshOverlay {
  constructor(canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(1, 1) : document.createElement('canvas')) {
    this.canvas = canvas;
    const gl = this.canvas.getContext('webgl2', { alpha: true, antialias: true, premultipliedAlpha: true, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is required for surface visibility.');
    this.gl = gl;
    try {
    const shader = (type, source) => {
      const item = gl.createShader(type); gl.shaderSource(item, source); gl.compileShader(item);
      if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
      return item;
    };
    const vs = shader(gl.VERTEX_SHADER, `#version 300 es
      in vec3 position;
      uniform vec3 translation;
      uniform vec2 viewport;
      uniform float focal;
      void main() {
        vec3 p = position + translation;
        float near = .01, far = 100.0;
        gl_Position = vec4(2.0*focal*p.x/viewport.x, -2.0*focal*p.y/viewport.y,
          (far+near)/(far-near)*p.z - 2.0*far*near/(far-near), p.z);
      }`);
    // clip: line in pixels from the top, fade length above it, canvas height.
    const fs = shader(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float;
      uniform vec4 color;
      uniform vec3 clip;
      out vec4 outColor;
      void main() {
        float keep = clamp((clip.x - (clip.z - gl_FragCoord.y)) / clip.y, 0.0, 1.0);
        outColor = vec4(color.rgb, color.a * keep);
      }`);
    this.program = gl.createProgram();
    gl.attachShader(this.program, vs); gl.attachShader(this.program, fs); gl.linkProgram(this.program);
    gl.deleteShader(vs); gl.deleteShader(fs);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
    this.vertices = gl.createBuffer(); this.faces = gl.createBuffer(); this.edges = gl.createBuffer();
    this.litFaces = gl.createBuffer(); this.litEdges = gl.createBuffer(); this.litKey = null;
    this.position = gl.getAttribLocation(this.program, 'position');
    this.uniforms = Object.fromEntries(['translation', 'viewport', 'focal', 'color', 'clip'].map(name => [name, gl.getUniformLocation(this.program, name)]));
    this.faceRef = null;
    } catch (error) { this.dispose(); throw error; }
  }

  /** mesh.faces only occlude (depth pass); `edges` are the drawn lines.
   * camera: source width/height/focal/translation. Output is width x height
   * pixels; alpha fades the whole wireframe; lines fade out towards `clipY`
   * (source pixels) and are not drawn below it. `highlight` {key, faces,
   * edges}: a body part with a mistake, tinted and outlined red. */
  draw(mesh, edges, camera, { width = camera.width, height = camera.height, alpha = 1, clipY = null, highlight = null } = {}) {
    const gl = this.gl;
    width = Math.max(1, Math.round(width)); height = Math.max(1, Math.round(height));
    if (this.canvas.width !== width) this.canvas.width = width;
    if (this.canvas.height !== height) this.canvas.height = height;
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertices); gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.position); gl.vertexAttribPointer(this.position, 3, gl.FLOAT, false, 0, 0);
    gl.uniform3fv(this.uniforms.translation, camera.translation);
    gl.uniform2f(this.uniforms.viewport, camera.width, camera.height);
    gl.uniform1f(this.uniforms.focal, camera.focal);
    const clip = Number.isFinite(clipY) ? clipY * height / camera.height : 4 * height;
    gl.uniform3f(this.uniforms.clip, clip, Math.max(4, height * 0.06), height);
    if (this.faceRef !== mesh.faces) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.faces); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.faces, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.edges); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, edges, gl.STATIC_DRAW);
      this.faceRef = mesh.faces;
    }
    gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL); gl.depthMask(true);
    gl.disable(gl.BLEND); gl.colorMask(false, false, false, false);
    gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(1, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.faces); gl.drawElements(gl.TRIANGLES, mesh.faces.length, gl.UNSIGNED_INT, 0);
    gl.disable(gl.POLYGON_OFFSET_FILL); gl.colorMask(true, true, true, true);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    const a = Math.max(0, Math.min(1, alpha));
    gl.depthMask(false); gl.uniform4f(this.uniforms.color, .82, 1, .97, .9 * a);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.edges); gl.drawElements(gl.LINES, edges.length, gl.UNSIGNED_INT, 0);
    if (highlight?.faces?.length) {
      if (this.litKey !== highlight.key) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.litFaces); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, highlight.faces, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.litEdges); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, highlight.edges, gl.DYNAMIC_DRAW);
        this.litKey = highlight.key;
      }
      // A translucent red skin over the part, then its lines in red on top.
      gl.uniform4f(this.uniforms.color, 1, .3, .37, .22 * a);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.litFaces); gl.drawElements(gl.TRIANGLES, highlight.faces.length, gl.UNSIGNED_INT, 0);
      gl.uniform4f(this.uniforms.color, 1, .3, .37, .95 * a);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.litEdges); gl.drawElements(gl.LINES, highlight.edges.length, gl.UNSIGNED_INT, 0);
    }
    gl.depthMask(true);
    return this.canvas;
  }

  dispose() {
    const gl = this.gl;
    if (this.vertices) gl.deleteBuffer(this.vertices);
    if (this.faces) gl.deleteBuffer(this.faces);
    if (this.edges) gl.deleteBuffer(this.edges);
    if (this.litFaces) gl.deleteBuffer(this.litFaces);
    if (this.litEdges) gl.deleteBuffer(this.litEdges);
    if (this.program) gl.deleteProgram(this.program);
    gl.getExtension('WEBGL_lose_context')?.loseContext();
  }
}
