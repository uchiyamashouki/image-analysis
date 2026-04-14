import { bootstrapApp } from "./src/main.js";

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    bootstrapApp();
  }, { once: true });
} else {
  bootstrapApp();
}
