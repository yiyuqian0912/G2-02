// Display-only ray casting of nearest sampled STORED labels. This shader
// never queries visibility and never generates a scientific response label.
import * as THREE from 'three';

export function responseVolume(labels, resolution) {
  const texture = new THREE.Data3DTexture(labels, resolution, resolution, resolution);
  texture.format = THREE.RedFormat;
  texture.type = THREE.UnsignedByteType;
  texture.minFilter = texture.magFilter = THREE.NearestFilter;
  texture.unpackAlignment = 1;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  const uniforms = {
    uVolume: {value:texture}, uResolution: {value:resolution}, uDensity: {value:1.8},
    uMode: {value:0}, uCropMin: {value:new THREE.Vector3(0,0,0)},
    uCropMax: {value:new THREE.Vector3(1,1,1)},
  };
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3, uniforms, side:THREE.BackSide, transparent:true,
    depthWrite:false, depthTest:false, toneMapped:false,
    vertexShader:`
      out vec3 vWorldPosition;
      void main() {
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * viewMatrix * worldPosition;
      }
    `,
    fragmentShader:`
      precision highp sampler3D;
      uniform sampler3D uVolume;
      uniform float uResolution;
      uniform float uDensity;
      uniform int uMode;
      uniform vec3 uCropMin;
      uniform vec3 uCropMax;
      in vec3 vWorldPosition;
      out vec4 volumeColor;

      int labelAt(vec3 p) {
        return int(floor(texture(uVolume, clamp(p, 0.0, 1.0)).r * 255.0 + 0.5));
      }
      float shadowAt(vec3 p) { return labelAt(p) == 0 ? 1.0 : 0.0; }
      void main() {
        vec3 direction = normalize(vWorldPosition - cameraPosition);
        vec3 inverseDirection = 1.0 / direction;
        vec3 a = (uCropMin - cameraPosition) * inverseDirection;
        vec3 b = (uCropMax - cameraPosition) * inverseDirection;
        vec3 nearAxis = min(a,b), farAxis = max(a,b);
        float nearT = max(max(nearAxis.x,nearAxis.y),nearAxis.z);
        float farT = min(min(farAxis.x,farAxis.y),farAxis.z);
        nearT = max(nearT,0.0);
        if (farT <= nearT) discard;
        float dt = 0.85 / uResolution;
        float t = nearT + 0.25 * dt;
        vec4 accumulated = vec4(0.0);
        int previous = -1;
        vec3 shadowColor = vec3(0.20,0.53,0.78);
        vec3 litColor = vec3(0.97,0.79,0.35);
        for (int i=0; i<800; ++i) {
          if (t > farT || accumulated.a > 0.97) break;
          vec3 p = cameraPosition + direction*t;
          int current = labelAt(p);
          if (uMode == 3) {
            if (current < 2 && previous >= 0 && previous < 2 && current != previous) {
              float h=1.0/uResolution;
              vec3 normal=vec3(
                shadowAt(p+vec3(h,0,0))-shadowAt(p-vec3(h,0,0)),
                shadowAt(p+vec3(0,h,0))-shadowAt(p-vec3(0,h,0)),
                shadowAt(p+vec3(0,0,h))-shadowAt(p-vec3(0,0,h)));
              if (length(normal)<0.01) normal=-direction;
              float shade=0.38+0.62*abs(dot(normalize(normal),normalize(vec3(0.5,-0.7,1.0))));
              accumulated=vec4(shadowColor*shade,0.87);
              break;
            }
          } else if (current < 2) {
            bool includeClass = uMode == 2 || (uMode == 0 && current == 0) || (uMode == 1 && current == 1);
            if (includeClass) {
              float density = uDensity * (uMode == 2 && current == 1 ? 0.3 : 1.0);
              float alpha = 1.0-exp(-density*dt);
              vec3 color = current == 0 ? shadowColor : litColor;
              accumulated.rgb += (1.0-accumulated.a)*alpha*color;
              accumulated.a += (1.0-accumulated.a)*alpha;
            }
          }
          previous=current;
          t+=dt;
        }
        if (accumulated.a < 0.005) discard;
        if (uMode != 3) accumulated.rgb /= max(accumulated.a,0.00001);
        volumeColor=accumulated;
      }
    `,
  });
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1,1,1),material);
  mesh.position.set(.5,.5,.5);
  mesh.renderOrder=2;
  mesh.userData.volumeTexture=texture;
  return mesh;
}
