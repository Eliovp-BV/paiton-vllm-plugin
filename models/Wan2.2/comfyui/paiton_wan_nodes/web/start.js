import { app } from "../../../scripts/app.js";
app.registerExtension({
  name: "Paiton.Wan.Welcome",
  async afterConfigureGraph() {
    const params = new URLSearchParams(location.search);
    const preset = params.get("preset") === "base" ? "base" : "fast";
    const key = `paiton.wan.started.${preset}`;
    if (!params.has("paiton") || localStorage.getItem(key)) return;
    localStorage.setItem(key, "1");
    try {
      await app.ui.settings.setSettingValue("Comfy.TutorialCompleted", true);
      const response = await fetch(`/api/workflow_templates/paiton_wan_nodes/${preset}.json`);
      if (!response.ok) throw new Error("Wan workflow unavailable");
      await app.loadGraphData(await response.json());
      app.canvas.ds.changeScale(0.9);
      app.canvas.ds.offset = [60, 140];
      app.canvas.setDirty(true, true);
    } catch (error) { localStorage.removeItem(key); console.error(error); }
  },
});
