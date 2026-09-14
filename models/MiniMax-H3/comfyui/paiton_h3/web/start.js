import { app } from "../../../scripts/app.js";

app.registerExtension({
  name: "Paiton.H3.Welcome",
  async afterConfigureGraph() {
    const params = new URLSearchParams(location.search);
    const key = params.has("studio") ? "paiton.h3.studio.started" : "paiton.h3.started";
    if (!params.has("paiton") || localStorage.getItem(key)) return;
    localStorage.setItem(key, "1");
    try {
      await app.ui.settings.setSettingValue("Comfy.TutorialCompleted", true);
      const preset = params.get("preset") === "turbo4" ? "turbo4" : "turbo8";
      const workflow = `${preset}-studio.json`;
      const response = await fetch(`/api/workflow_templates/paiton_h3/${workflow}`);
      if (!response.ok) throw new Error("Paiton workflow unavailable");
      await app.loadGraphData(await response.json());
      const focus = app.graph._nodes.find(node => node.type === "PaitonH3VideoSettings") ||
                    app.graph._nodes.find(node => node.type === "MiniMaxH3ImageToVideo");
      if (focus) {
        app.canvas.ds.changeScale(0.75);
        app.canvas.centerOnNode(focus);
      }
    } catch (error) {
      localStorage.removeItem(key);
      console.error(error);
    }
  },
});

app.registerExtension({
  name: "Paiton.H3.VideoSettings",
  nodeCreated(node) {
    if (node.comfyClass !== "PaitonH3VideoSettings") return;
    const find = (name) => node.widgets.find(widget => widget.name === name);
    const update = () => {
      const sizes = [[576, 320], [704, 384], [864, 480]];
      let [width, height] = sizes[Number(find("resolution").value) - 1] || sizes[2];
      if (find("aspect").value === "Portrait") [width, height] = [height, width];
      if (find("aspect").value === "Square") width = height;
      const requested = Number(find("duration_seconds").value) * 24;
      const frames = requested + ((5 - requested) % 17 + 17) % 17;
      node.title = `${width} × ${height} · ${(frames / 24).toFixed(2)} s · ${frames} frames`;
      node.setDirtyCanvas(true, true);
    };
    for (const name of ["duration_seconds", "resolution", "aspect"]) {
      const widget = find(name);
      const previous = widget.callback;
      widget.callback = function (...args) { previous?.apply(this, args); update(); };
    }
    const previous = node.onConfigure;
    node.onConfigure = function (...args) { previous?.apply(this, args); update(); };
    update();
  },
});
