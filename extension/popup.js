const API_URL = "http://127.0.0.1:8000/predict";
const CONTENT_API_URL = "http://127.0.0.1:8000/predict_html";

const urlElement = document.querySelector("#url");
const checkButton = document.querySelector("#check");
const contentButton = document.querySelector("#check-content");
const resultElement = document.querySelector("#result");
const riskBadge = document.querySelector("#risk-badge");
const probabilityElement = document.querySelector("#probability");
const reasonsElement = document.querySelector("#reasons");
const errorElement = document.querySelector("#error");
const modelElement = document.querySelector("#model-info");

let activeUrl = "";
let activeTabId;

async function readActiveUrl() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  activeUrl = tab?.url ?? "";
  activeTabId = tab?.id;
  urlElement.textContent = activeUrl || "No active HTTP page was found.";
  checkButton.disabled = !/^https?:\/\//i.test(activeUrl);
  contentButton.disabled = checkButton.disabled;
}

function showError(message) {
  resultElement.classList.add("hidden");
  errorElement.textContent = message;
  errorElement.classList.remove("hidden");
}

function showResult(result) {
  errorElement.classList.add("hidden");
  riskBadge.textContent = `${result.risk_level} risk`;
  riskBadge.className = `badge ${result.risk_level}`;
  probabilityElement.textContent = `Model risk score: ${(result.phishing_probability * 100).toFixed(1)}/100`;
  modelElement.textContent = `${result.experimental ? "Experimental · " : ""}${result.model_type ?? "URL model"}${result.feature_count ? ` · ${result.feature_count} features` : ""} · ${result.model_sha256?.slice(0, 12) ?? result.model_version} · cutoff ${result.threshold.toFixed(4)}`;
  reasonsElement.replaceChildren(
    ...result.reasons.map((reason) => {
      const item = document.createElement("li");
      item.textContent = reason;
      return item;
    }),
  );
  resultElement.classList.remove("hidden");
}

async function readPageHtml() {
  // activeTab grants one-off access to the page the user is checking.
  const [injection] = await chrome.scripting.executeScript({
    target: { tabId: activeTabId },
    func: () => document.documentElement.outerHTML,
  });
  return injection.result;
}

async function runCheck(button, endpoint, buildBody) {
  const label = button.textContent;
  checkButton.disabled = true;
  contentButton.disabled = true;
  button.textContent = "Checking…";
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(await buildBody()),
    });
    const body = await response.json();
    if (!response.ok) {
      const message = Array.isArray(body.detail)
        ? body.detail.map((item) => item.msg).join("; ")
        : body.detail;
      throw new Error(message ?? "The API rejected this URL.");
    }
    showResult(body);
  } catch (error) {
    showError(
      error instanceof TypeError
        ? "Cannot reach the local API. Start Uvicorn on port 8000."
        : error.message,
    );
  } finally {
    checkButton.disabled = false;
    contentButton.disabled = false;
    button.textContent = label;
  }
}

checkButton.addEventListener("click", () =>
  runCheck(checkButton, API_URL, () => ({ url: activeUrl })),
);

contentButton.addEventListener("click", () =>
  runCheck(contentButton, CONTENT_API_URL, async () => ({
    url: activeUrl,
    html: await readPageHtml(),
  })),
);

readActiveUrl().catch(() => showError("Could not read the active tab URL."));
