// static/js/ai_chat.js

console.log("AI Chat JS Loaded");

// Track selected sample + meta
let selectedSampleIdx = null;
let sampleMeta = [];
let lastUserMessage = "";

// ===========================
// Load Grid Metadata
// ===========================
async function loadGridMeta() {
    try {
        const res = await fetch('/ai_chat/meta');
        const data = await res.json();

        if (data.ok && data.meta && data.meta.cells) {
            sampleMeta = data.meta.cells;
            console.log("✓ Loaded sample metadata:", sampleMeta.length);
            renderSampleSelectors();
        } else {
            console.warn("⚠ No meta found for grid cells");
        }
    } catch (err) {
        console.error("Error loading grid metadata:", err);
    }
}

// ===========================
// Render Sample Buttons
// ===========================
function renderSampleSelectors() {
    const container = document.getElementById("gridCellsList");  // Correct HTML ID
    
    if (!container) {
        console.error("❌ gridCellsList container not found");
        return;
    }

    container.innerHTML = ""; // Clear existing

    sampleMeta.forEach((cell, idx) => {
        const btn = document.createElement("button");
        btn.className = "btn btn-outline-info btn-sm m-1 sample-btn";
        btn.textContent = `Sample ${idx}`;
        btn.addEventListener("click", () => selectSample(idx));
        container.appendChild(btn);
    });

    console.log("✓ Rendered sample selector buttons");
}

// ===========================
// Handle Sample Selection
// ===========================
function selectSample(idx) {
    selectedSampleIdx = idx;
    console.log("Selected Sample:", idx);

    // Highlight active selected button
    document.querySelectorAll(".sample-btn").forEach(btn => btn.classList.remove("active"));
    const activeBtn = document.querySelectorAll(".sample-btn")[idx];
    if (activeBtn) activeBtn.classList.add("active");

    // Update details card
    const cell = sampleMeta[idx];
    const sel = document.getElementById("selectedSample");
    const det = document.getElementById("selectedDetails");

    if (!cell) return;

    if (sel) sel.textContent = `Sample ${idx} Selected`;

    if (det) {
        det.innerHTML = `
            <div>True: <strong>${cell.true}</strong></div>
            <div>Predicted: <strong>${cell.pred}</strong></div>
            <div style="color:${cell.true === cell.pred ? 'lightgreen' : 'red'};">
                ${cell.true === cell.pred ? "Correct ✔" : "Incorrect ✖"}
            </div>`;
    }

    const chatBox = document.getElementById("chatBox");
    if (chatBox) chatBox.scrollTop = chatBox.scrollHeight;
}

// ===========================
// Chat UI Helper
// ===========================
function addMessage(text, isUser = false) {
    const box = document.getElementById("chatBox");
    if (!box) return;

    const msg = document.createElement("div");
    msg.className = `chat-message ${isUser ? "chat-user" : "chat-assistant"}`;
    msg.textContent = text;

    box.appendChild(msg);
    box.scrollTop = box.scrollHeight;
}

// ===========================
// Normal API Chat
// ===========================
async function sendAsk(msg) {
    const payload = {
        message: msg,
        sample_index: selectedSampleIdx
    };

    const response = await fetch("/ai_chat/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    if (data.ok && data.reply) {
        addMessage(data.reply, false);
        saveHistory(msg, data.reply);
    }
}

// ===========================
// Streaming Chat API
// ===========================
async function sendStream(msg) {
    const payload = {
        message: msg,
        sample_index: selectedSampleIdx
    };

    try {
        const res = await fetch("/ai_chat/stream", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            // Fallback to non-streaming if streaming fails
            console.warn("Streaming failed, using fallback");
            await sendAsk(msg);
            return;
        }

        const reader = res.body.getReader();
        let fullText = "";
        const decoder = new TextDecoder('utf-8');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            // Remove non-ASCII characters on client side too
            const cleanChunk = chunk.replace(/[^\x00-\x7F]/g, '');
            fullText += cleanChunk;
            updateStreaming(fullText);
        }

        finalizeStream(fullText);
    } catch (error) {
        console.error("Stream error:", error);
        addMessage("Sorry, I encountered an error. Please check if GROQ_API_KEY is set in your environment.", false);
    }
}

function updateStreaming(text) {
    const box = document.getElementById("chatBox");
    let last = box.lastElementChild;

    if (!last || !last.classList.contains("chat-assistant")) {
        last = document.createElement("div");
        last.className = "chat-message chat-assistant";
        box.appendChild(last);
    }

    last.textContent = text;
    box.scrollTop = box.scrollHeight;
}

function finalizeStream(text) {
    saveHistory(lastUserMessage, text);
}

// ===========================
// Save chat to server
// ===========================
async function saveHistory(user, assistant) {
    await fetch("/ai_chat/save_message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            user_message: user,
            assistant_message: assistant,
            sample_index: selectedSampleIdx
        })
    });
}

// ===========================
// Chat Input Setup
// ===========================
function setupChat() {
    const sendBtn = document.getElementById("chatSend");
    const input = document.getElementById("chatInput");

    if (!sendBtn || !input) return;

    sendBtn.addEventListener("click", () => {
        const msg = input.value.trim();
        if (!msg) return;

        addMessage(msg, true);
        lastUserMessage = msg;
        input.value = "";

        // Always use streaming for all questions
        sendStream(msg);
    });

    input.addEventListener("keypress", (e) => {
        if (e.key === "Enter") sendBtn.click();
    });
}

// ===========================
// On Page Load
// ===========================
document.addEventListener("DOMContentLoaded", () => {
    console.log("✓ AI Chat Ready");
    setupChat();
    loadGridMeta();
});
