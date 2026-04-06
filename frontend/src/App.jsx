import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

const api = axios.create({
  baseURL: "",
  timeout: 120000,
});

const LEGEND = [
  { name: "background", color: "#000000" },
  { name: "trees", color: "#228B22" },
  { name: "logs", color: "#8B4513" },
  { name: "rocks", color: "#808080" },
  { name: "flowers", color: "#FF69B4" },
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [maskSrc, setMaskSrc] = useState("");
  const [originalPreview, setOriginalPreview] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [iou, setIou] = useState(null);
  const [history, setHistory] = useState([]);
  const [beforeAfter, setBeforeAfter] = useState(null);
  const [chartBust, setChartBust] = useState(0);

  const refreshMetrics = useCallback(async () => {
    try {
      const { data } = await api.get("/metrics");
      setHistory(data.history || []);
      setChartBust((x) => x + 1);
    } catch (e) {
      console.warn(e);
    }
    try {
      const { data } = await api.get("/before-after");
      setBeforeAfter(data);
    } catch {
      setBeforeAfter(null);
    }
  }, []);

  useEffect(() => {
    api
      .get("/health")
      .then(({ data }) => setHealth(data))
      .catch(() => setHealth({ status: "offline" }));
    refreshMetrics();
    const id = setInterval(refreshMetrics, 8000);
    return () => clearInterval(id);
  }, [refreshMetrics]);

  const onFile = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError("");
    setMaskSrc("");
    setIou(null);
    setOriginalPreview(URL.createObjectURL(file));
    const fd = new FormData();
    fd.append("image", file);
    try {
      const { data } = await api.post("/predict", fd, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMaskSrc(`data:image/png;base64,${data.mask_png_base64}`);
      setConfidence(data.mean_confidence);
      if (data.iou_mean != null) setIou(data.iou_mean);
    } catch (err) {
      setError(err.response?.data?.error || err.message || "predict failed");
    }
  };

  const startTrain = async () => {
    setBusy("training");
    setError("");
    try {
      await api.post("/train", { epochs: 6, model: "segformer" });
    } catch (err) {
      setError(err.message);
    }
    setTimeout(() => setBusy(""), 1500);
  };

  const startSelfTrain = async () => {
    setBusy("self-training");
    setError("");
    try {
      await api.post("/self-train", { epochs: 4 });
    } catch (err) {
      setError(err.message);
    }
    setTimeout(() => setBusy(""), 1500);
  };

  const reloadModel = async () => {
    try {
      await api.post("/reload-model");
      await refreshMetrics();
    } catch (e) {
      setError(e.message);
    }
  };

  const chartData = history
    .filter((h) => h.epoch != null)
    .map((h) => ({
      epoch: h.epoch,
      val_mIoU: h.val_mIoU,
      train_loss: h.train_loss,
      phase: h.phase || "train",
    }));

  return (
    <>
      <div className="badge">Hackathon demo · SegFormer + domain randomization</div>
      <h1>Domain-Adaptive Self-Improving Segmentation</h1>
      <p className="sub">
        Upload a desert-style scene. The API returns a colored mask (trees, logs, rocks,
        flowers). Use <strong>Train</strong> for baseline learning and{" "}
        <strong>Improve model</strong> to run pseudo-labeling on{" "}
        <code style={{ fontFamily: "var(--mono)", fontSize: "0.85em" }}>data/unlabeled</code>{" "}
        and fine-tune.
      </p>

      <div className="stats">
        API: {health?.status === "ok" ? "● online" : "○ check Flask"} · device:{" "}
        {health?.device || "—"}
        {busy && ` · ${busy}…`}
      </div>

      <div className="grid grid-2" style={{ marginTop: "1.25rem" }}>
        <div className="panel">
          <h2>Segmentation</h2>
          <input type="file" accept="image/*" onChange={onFile} />
          <div className="btn-row">
            <button type="button" className="btn-primary" onClick={startTrain} disabled={!!busy}>
              Train baseline
            </button>
            <button type="button" className="btn-secondary" onClick={startSelfTrain} disabled={!!busy}>
              Improve model (self-train)
            </button>
            <button type="button" className="btn-secondary" onClick={reloadModel}>
              Reload weights
            </button>
          </div>
          {error && <p className="error">{error}</p>}
          <div className="preview grid grid-2">
            <div>
              <div className="stats">Input</div>
              {originalPreview ? (
                <img src={originalPreview} alt="input" />
              ) : (
                <p className="stats">No image yet</p>
              )}
            </div>
            <div>
              <div className="stats">Predicted mask</div>
              {maskSrc ? (
                <img src={maskSrc} alt="mask" />
              ) : (
                <p className="stats">Run predict</p>
              )}
            </div>
          </div>
          {confidence != null && (
            <p className="stats">Mean pixel confidence: {(confidence * 100).toFixed(1)}%</p>
          )}
          {iou != null && (
            <p className="stats">IoU vs uploaded GT mask: {(iou * 100).toFixed(2)}%</p>
          )}
          <div className="legend">
            {LEGEND.map((l) => (
              <span key={l.name}>
                <span className="swatch" style={{ background: l.color }} />
                {l.name}
              </span>
            ))}
          </div>
        </div>

        <div className="panel">
          <h2>Before vs after (self-training)</h2>
          {beforeAfter && (beforeAfter.before_mIoU != null || beforeAfter.after_mIoU != null) ? (
            <div className="stats">
              <p>
                Validation mIoU before:{" "}
                <strong>{beforeAfter.before_mIoU?.toFixed(4) ?? "—"}</strong>
              </p>
              <p>
                After self-train: <strong>{beforeAfter.after_mIoU?.toFixed(4) ?? "—"}</strong>
              </p>
            </div>
          ) : (
            <p className="stats">{beforeAfter?.note || "Run self-training to populate metrics."}</p>
          )}

          <h2 style={{ marginTop: "1.25rem" }}>Training curves</h2>
          <p className="stats" style={{ marginBottom: "0.5rem" }}>
            Loss / mIoU from <code>logs/training_history.jsonl</code>
          </p>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <XAxis dataKey="epoch" stroke="#8fa3b8" />
                <YAxis stroke="#8fa3b8" />
                <Tooltip
                  contentStyle={{ background: "#1a222d", border: "1px solid #2d3a4d" }}
                />
                <Legend />
                <Line type="monotone" dataKey="val_mIoU" stroke="#3ddc97" dot={false} name="val mIoU" />
                <Line type="monotone" dataKey="train_loss" stroke="#ffb347" dot={false} name="train loss" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <p className="stats">No history yet — train the model first.</p>
          )}

          <h2 style={{ marginTop: "1rem" }}>Saved chart (matplotlib)</h2>
          <img
            src={`/charts/training_segformer.png?bust=${chartBust}`}
            alt="training chart"
            style={{ maxWidth: "100%", borderRadius: 8, border: "1px solid var(--border)" }}
            onError={(e) => {
              e.target.style.display = "none";
            }}
          />
        </div>
      </div>
    </>
  );
}
