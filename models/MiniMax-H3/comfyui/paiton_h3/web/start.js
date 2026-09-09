import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "Paiton.H3.Welcome",
  async afterConfigureGraph() {
    if (!new URLSearchParams(location.search).has("paiton") ||
        localStorage.getItem("paiton.h3.started")) return;
    localStorage.setItem("paiton.h3.started", "1");
    try {
      await app.ui.settings.setSettingValue("Comfy.TutorialCompleted", true);
      const preset = new URLSearchParams(location.search).get("preset") === "turbo4" ? "turbo4" : "turbo8";
      const workflow = preset === "turbo4" ? "turbo4.json" : "turbo8-15s.json";
      const response = await fetch(`/api/workflow_templates/paiton_h3/${workflow}`);
      if (!response.ok) throw new Error("Paiton workflow unavailable");
      await app.loadGraphData(await response.json());
      const prompt = app.graph._nodes.find(node => node.type === "MiniMaxH3ImageToVideo");
      if (prompt) {
        app.canvas.ds.changeScale(0.85);
        app.canvas.centerOnNode(prompt);
      }
    } catch (error) {
      localStorage.removeItem("paiton.h3.started");
      console.error(error);
    }
  },
});
