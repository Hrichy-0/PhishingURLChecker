const API_URL = "http://127.0.0.1:8000/predict";

const urlElement = document.querySelector("#url");
const checkButton = document.querySelector("#check");
const resultElement = document.querySelector("#result");
const riskBadge = document.querySelector("#risk-badge");
const probabilityElement = document.querySelector("#probability");
const reasonsElement = document.querySelector("#reasons");
const errorElement = document.querySelector("#error");

let activeUrl = "";

async function readActiveUrl() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  activeUrl = tab?.url ?? "";
  urlElement.textContent = activeUrl || "No active HTTP page was found.";
  checkButton.disabled = !/^https?:\/\//i.test(activeUrl);
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
  probabilityElement.textContent = `${(result.phishing_probability * 100).toFixed(1)}% phishing`;
  reasonsElement.replaceChildren(
    ...result.reasons.map((reason) => {
      const item = document.createElement("li");
      item.textContent = reason;
      return item;
    }),
  );
  resultElement.classList.remove("hidden");
}

checkButton.addEventListener("click", async () => {
  checkButton.disabled = true;
  checkButton.textContent = "Checking…";
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: activeUrl }),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail ?? "The API rejected this URL.");
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
    checkButton.textContent = "Check this URL";
  }
});

readActiveUrl().catch(() => showError("Could not read the active tab URL."));
