import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "Paiton.Flux2.Welcome",
  async afterConfigureGraph() {
    if (!new URLSearchParams(location.search).has("paiton") ||
        localStorage.getItem("paiton.flux2.started")) return;
    localStorage.setItem("paiton.flux2.started", "1");
    try {
      await app.ui.settings.setSettingValue("Comfy.TutorialCompleted", true);
      const response = await fetch("/api/workflow_templates/paiton_flux2/Generate.json");
      if (!response.ok) throw new Error("Paiton workflow unavailable");
      await app.loadGraphData(await response.json());
    } catch (error) {
      localStorage.removeItem("paiton.flux2.started");
      console.error(error);
    }
  },
});
