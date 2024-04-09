const coords = { x: 0, y: 0 };
const circles = document.querySelectorAll(".circle");
const colors = [
  "#1f005c",
  "#1d146c",
  "#19247c",
  "#12348b",
  "#044299",
  "#0051a7",
  "#0060b5",
  "#006fc2",
  "#007ece",
  "#008dda",
  "#009ce5",
  "#00abf0",
];
circles.forEach(function (circle, index) {
  circle.x = 0;
  circle.y = 0;
  circle.style.backgroundColor = colors[index % colors.length];
});

window.addEventListener("mousemove", function (e) {
  coords.x = e.clientX;
  coords.y = e.clientY;
});

function animateCircles() {
  let x = coords.x;
  let y = coords.y;

  circles.forEach(function (circle, index) {
    circle.style.left = x - 12 + "px";
    circle.style.top = y - 12 + "px";

    circle.style.scale = 1 - index / circles.length;

    circle.x = x;
    circle.y = y;

    const nextCirlce = circles[index + 1] || circles[0];
    x += (nextCirlce.x - x) * 0.2;
    y += (nextCirlce.y - y) * 0.2;
  });
  requestAnimationFrame(animateCircles);
}
animateCircles();
