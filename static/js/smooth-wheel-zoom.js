// Wheel zoom shared by the point-cloud and mesh viewers. TrackballControls
// interprets each wheel event as a position jump and applies frame-based damping;
// trackpads and high-resolution wheels can therefore feel uneven.
export function createSmoothWheelZoom(canvas, camera, controls) {
  let goal = null;
  const distance = () => camera.position.distanceTo(controls.target);
  canvas.addEventListener('wheel', event => {
    if (!controls.enabled) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    const pixels = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? canvas.clientHeight : 1);
    if (!Number.isFinite(pixels) || !pixels) return;
    const base = goal ?? distance();
    goal = Math.max(camera.near * 5, Math.min(camera.far / 4,
      base * Math.exp(Math.max(-240, Math.min(240, pixels)) * 0.0015)));
  }, {capture: true, passive: false});
  return {
    reset() { goal = null; },
    update(dt) {
      if (goal === null) return;
      const current = distance();
      if (Math.abs(current - goal) < Math.max(goal * 0.0001, camera.near * 0.01)) {
        goal = null;
        return;
      }
      const next = current + (goal - current) * (1 - Math.exp(-Math.min(dt, 0.1) * 14));
      camera.position.sub(controls.target).multiplyScalar(next / current).add(controls.target);
    }
  };
}
