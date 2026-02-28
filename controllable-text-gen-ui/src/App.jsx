import { useState } from "react";
import "./App.css";

const STYLES = [
  {
    id: "formal",
    label: "Formal",
    emoji: "🧑‍💼",
    description: "Professional and polished",
  },
  {
    id: "casual",
    label: "Casual",
    emoji: "😌",
    description: "Chatty and relaxed",
  },
  {
    id: "enthusiastic",
    label: "Enthusiastic",
    emoji: "🤩",
    description: "Lively and energetic",
  },
  {
    id: "sarcastic",
    label: "Sarcastic",
    emoji: "😏",
    description: "Snarky and ironic",
  },
  {
    id: "poetic",
    label: "Poetic",
    emoji: "🌙",
    description: "Descriptive and expressive",
  },
  {
    id: "neutral",
    label: "Neutral",
    emoji: "😐",
    description: "Plain and balanced",
  },
];

function App() {
  const [text, setText] = useState("");
  const [style, setStyle] = useState("formal");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastRequest, setLastRequest] = useState({ text: "", style: "" });
  const [compareResults, setCompareResults] = useState(null);
  const [strength, setStrength] = useState(70); // default medium-strong
  const lastStyleMeta =
    lastRequest.style && lastRequest.style !== "multi"
      ? STYLES.find((s) => s.id === lastRequest.style)
      : null;

  async function handleGenerate(e) {
    e.preventDefault();
    setError("");
    setResult("");
    setCompareResults(null);

    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please enter some text first.");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed, style, strength }),
      });

      if (!res.ok) {
        throw new Error("Server error");
      }

      const data = await res.json();
      setResult(data.result);
      setLastRequest({ text: trimmed, style });
    } catch (err) {
      console.error(err);
      setError("Could not contact the model server. Is it running?");
    } finally {
      setLoading(false);
    }
  }

  async function handleCompareAll() {
    setError("");
    setResult("");

    const trimmed = text.trim();
    if (!trimmed) {
      setError("Please enter some text first.");
      return;
    }

    setCompareLoading(true);
    try {
      const res = await fetch("http://127.0.0.1:8000/generate-multi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed, strength }),
      });

      if (!res.ok) {
        throw new Error("Server error");
      }

      const data = await res.json();
      setCompareResults(data.results);
      setLastRequest({ text: trimmed, style: "multi" });
    } catch (err) {
      console.error(err);
      setError("Could not contact the model server. Is it running?");
    } finally {
      setCompareLoading(false);
    }
  }

  const activeStyleMeta = STYLES.find((s) => s.id === style);

  return (
    <div className="ctg-root">
      <div className={`ctg-shell ctg-theme-${style}`}>
        <header className="ctg-header">
          <div>
            <h1>Controllable Text Studio</h1>
            <p>
              Explore how the same sentence changes when you steer a language
              model&apos;s tone.
            </p>
          </div>
        </header>

        <main className="ctg-main">
          <section className="ctg-input-panel">
            <form onSubmit={handleGenerate} className="ctg-form">
              <label className="ctg-label">
                Input text
                <span className="ctg-label-hint">
                  Try a review, opinion, or short explanation.
                </span>
              </label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={5}
                className="ctg-textarea"
                placeholder='Example: "I appreciate the effort, but the result was not what I expected."'
              />

              <div className="ctg-tone-row">
                <div className="ctg-tone-list-label">
                  Tone
                  <span className="ctg-label-hint">
                    Click a chip to choose how the model should sound.
                  </span>
                </div>
                <div className="ctg-tone-chips">
                  {STYLES.map((s) => (
                    <button
                      type="button"
                      key={s.id}
                      className={
                        "ctg-tone-chip" +
                        (s.id === style ? " ctg-tone-chip--active" : "")
                      }
                      onClick={() => setStyle(s.id)}
                    >
                      <div className="ctg-tone-chip-main">
                        <span className={`ctg-tone-orb ctg-tone-orb-${s.id}`} />
                        <div className="ctg-tone-chip-text">
                          <span className="ctg-tone-chip-label">
                            <span className="ctg-tone-chip-emoji">
                              {s.emoji}
                            </span>
                            {s.label}
                          </span>
                          <span className="ctg-tone-chip-desc">
                            {s.description}
                          </span>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="ctg-strength-row">
                <div className="ctg-strength-label">
                  Style strength
                  <span className="ctg-label-hint">
                    Controls how strong or subtle the tone is.
                  </span>
                </div>
                <div className="ctg-strength-control">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={strength}
                    onChange={(e) => setStrength(Number(e.target.value))}
                    className="ctg-strength-slider"
                  />
                  <div className="ctg-strength-value">
                    {strength <= 20 && "Very subtle"}
                    {strength > 20 && strength <= 60 && "Balanced"}
                    {strength > 60 && "Very strong"}
                    <span className="ctg-strength-number">({strength})</span>
                  </div>
                </div>
              </div>

              <div className="ctg-actions">
                <button
                  type="submit"
                  className="ctg-button"
                  disabled={loading || compareLoading}
                >
                  {loading ? "Generating..." : "Generate rewrite"}
                </button>

                <button
                  type="button"
                  className="ctg-button ctg-button-secondary"
                  onClick={handleCompareAll}
                  disabled={compareLoading || loading}
                >
                  {compareLoading ? "Comparing..." : "Compare all tones"}
                </button>

                {activeStyleMeta && (
                  <div className="ctg-active-style-pill">
                    Current tone:{" "}
                    <strong>{activeStyleMeta.label.toLowerCase()}</strong>
                  </div>
                )}
              </div>
            </form>

            {error && <div className="ctg-error">{error}</div>}
          </section>

          <section className="ctg-output-panel">
            <div className="ctg-output-card">
              <div className="ctg-output-header">
                <div>
                  <h2>Styled output</h2>
                </div>
                {lastRequest.text &&
                  lastRequest.style !== "multi" &&
                  lastStyleMeta && (
                    <div className="ctg-badge">
                      <span className="ctg-badge-emoji">
                        {lastStyleMeta.emoji}
                      </span>
                      {lastStyleMeta.label} tone
                    </div>
                  )}
              </div>

              {!result && !compareResults && !loading && !compareLoading && (
                <div className="ctg-placeholder">
                  <p>
                    Your rewrite will appear here. Enter some text and generate
                    a single tone or compare all tones side by side.
                  </p>
                </div>
              )}

              {(loading || compareLoading) && (
                <div className="ctg-skeleton">
                  <div className="ctg-skeleton-line" />
                  <div className="ctg-skeleton-line" />
                  <div className="ctg-skeleton-line short" />
                </div>
              )}

              {result && !loading && !compareResults && (
                <div className="ctg-output-body">
                  <div className="ctg-output-block">
                    <div className="ctg-output-label">Original</div>
                    <p className="ctg-output-text ctg-output-text--muted">
                      {lastRequest.text}
                    </p>
                  </div>

                  <div className="ctg-output-block">
                    <div className="ctg-output-label">Rewritten</div>
                    <p className="ctg-output-text">{result}</p>
                  </div>
                </div>
              )}

              {compareResults && !compareLoading && (
                <div className="ctg-compare-grid">
                  {STYLES.map((s) => (
                    <div key={s.id} className="ctg-compare-card">
                      <div className="ctg-compare-header">
                        <div className="ctg-compare-title">{s.label}</div>
                        <div className="ctg-compare-subtitle">
                          {s.description}
                        </div>
                      </div>
                      <p className="ctg-compare-text">{compareResults[s.id]}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>
        </main>

        <footer className="ctg-footer">
          <p>Made with ❤️ by Raksha Pawar &nbsp;|&nbsp; &copy; 2025 </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
