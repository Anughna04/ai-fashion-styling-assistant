const API_BASE = "http://127.0.0.1:8000";

// State
let authToken = localStorage.getItem("token");
let userPrefs = JSON.parse(localStorage.getItem("prefs") || "{}");
let imageFile = null;

// DOM Elements
const authView = document.getElementById("auth-view");
const dashboardView = document.getElementById("dashboard-view");
const loginFormBox = document.getElementById("login-form-box");
const signupFormBox = document.getElementById("signup-form-box");

// 1. Init
document.addEventListener("DOMContentLoaded", () => {
    if (authToken) {
        showDashboard();
    } else {
        showAuth();
    }
});

// 2. View Switching
async function showDashboard() {
    authView.classList.add("hidden");
    dashboardView.classList.remove("hidden");
    
    // Fetch User Stats
    try {
        const response = await fetch(`${API_BASE}/user-stats`, {
            headers: { "Authorization": `Bearer ${authToken}` }
        });
        if (response.ok) {
            const data = await response.json();
            document.getElementById("welcome-user").textContent = `Welcome back, ${data.name} 👋`;
            document.getElementById("stat-saved").textContent = data.saved_count;
            document.getElementById("stat-recs").textContent = data.recommendation_count;
        } else if (response.status === 401 || response.status === 403) {
            // Token expired or invalid
            document.getElementById("logout-btn").click();
            return;
        }
    } catch (e) { console.error("Stats fetch failed", e); }

    // Fetch Recent Activity
    fetchRecentActivity();
}
function showAuth() {
    authView.classList.remove("hidden");
    dashboardView.classList.add("hidden");
}

document.getElementById("go-to-signup").addEventListener("click", (e) => {
    e.preventDefault();
    loginFormBox.classList.add("hidden");
    signupFormBox.classList.remove("hidden");
});
document.getElementById("go-to-login").addEventListener("click", (e) => {
    e.preventDefault();
    signupFormBox.classList.add("hidden");
    loginFormBox.classList.remove("hidden");
});

document.getElementById("logout-btn").addEventListener("click", (e) => {
    if(e) e.preventDefault();
    localStorage.removeItem("token");
    localStorage.removeItem("prefs");
    authToken = null;
    showAuth();
});

// 3. Multistep Signup
const signupSteps = document.querySelectorAll(".signup-step");
const currentStepIndicator = document.getElementById("current-step");

document.querySelectorAll(".next-step").forEach(btn => {
    btn.addEventListener("click", (e) => {
        let target = btn.getAttribute("data-target");
        switchStep(target);
    });
});
document.querySelectorAll(".prev-step").forEach(btn => {
    btn.addEventListener("click", (e) => {
        let target = btn.getAttribute("data-target");
        switchStep(target);
    });
});
function switchStep(target) {
    signupSteps.forEach(s => s.classList.remove("active"));
    document.getElementById(`signup-step-${target}`).classList.add("active");
    currentStepIndicator.textContent = target;
}

// Chips Logic
const chips = document.querySelectorAll(".chip");
let selectedStyles = new Set();
chips.forEach(chip => {
    chip.addEventListener("click", () => {
        const val = chip.getAttribute("data-val");
        if (selectedStyles.has(val)) {
            selectedStyles.delete(val);
            chip.classList.remove("selected");
        } else {
            selectedStyles.add(val);
            chip.classList.add("selected");
        }
    });
});

// Auth API Calls
document.getElementById("login-btn").addEventListener("click", async () => {
    const email = document.getElementById("login-email").value;
    const password = document.getElementById("login-password").value;
    
    try {
        const response = await fetch(`${API_BASE}/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email, password })
        });
        const data = await response.json();
        if (response.ok) {
            localStorage.setItem("token", data.access_token);
            localStorage.setItem("prefs", JSON.stringify(data.preferences));
            authToken = data.access_token;
            showDashboard();
        } else alert(data.detail);
    } catch (e) {
        console.error(e);
        alert("Login failed.");
    }
});

document.getElementById("complete-signup-btn").addEventListener("click", async () => {
    const name = document.getElementById("signup-name").value;
    const email = document.getElementById("signup-email").value;
    const password = document.getElementById("signup-password").value;
    const gender = document.getElementById("signup-gender").value;
    const style_type = document.getElementById("signup-style-type").value;
    
    const preferences = {
        gender,
        style_type,
        styles: Array.from(selectedStyles)
    };

    try {
        const response = await fetch(`${API_BASE}/signup`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, email, password, preferences })
        });
        const data = await response.json();
        if (response.ok) {
            localStorage.setItem("token", data.access_token);
            localStorage.setItem("prefs", JSON.stringify(preferences));
            authToken = data.access_token;
            showDashboard();
        } else alert(data.detail);
    } catch (e) {
        alert("Signup failed.");
    }
});

// 4. Tab Logic & Suggestions
const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        const tabId = btn.getAttribute("data-tab");
        switchTab(tabId);
    });
});

function switchTab(tabId) {
    tabBtns.forEach(b => b.classList.toggle("active", b.getAttribute("data-tab") === tabId));
    tabContents.forEach(c => c.classList.toggle("active", c.id === `tab-${tabId}`));
}

window.setQuery = function(text) {
    document.getElementById("text-query").value = text;
    switchTab("describe");
};

async function fetchRecentActivity() {
    const list = document.getElementById("activity-list");
    try {
        const response = await fetch(`${API_BASE}/history`, {
            headers: { "Authorization": `Bearer ${authToken}` }
        });
        const data = await response.json();
        if(data && data.length > 0) {
            list.innerHTML = "";
            data.slice(0, 3).forEach(h => {
                const timeStr = new Date(h.time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
                list.innerHTML += `
                    <div class="activity-item">
                        <div class="activity-icon"><i class="fa fa-sparkles"></i></div>
                        <div class="activity-info">
                            <p class="activity-title">${h.prompt || "Visual Search"}</p>
                            <p class="activity-time">${timeStr}</p>
                        </div>
                    </div>
                `;
            });
        }
    } catch(e) { console.error(e); }
}

// 5. Drag & Drop Image
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const imagePreview = document.getElementById("image-preview");
const clearBtn = document.getElementById("clear-image");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("active"); });
dropZone.addEventListener("dragleave", () => { dropZone.classList.remove("active"); });
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("active");
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});
fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
    imageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove("hidden");
        clearBtn.classList.remove("hidden");
        dropZone.querySelector("i").classList.add("hidden");
        dropZone.querySelector("p").classList.add("hidden");
        dropZone.querySelector("small").classList.add("hidden");
    };
    reader.readAsDataURL(file);
}

clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    imageFile = null;
    fileInput.value = "";
    imagePreview.classList.add("hidden");
    clearBtn.classList.add("hidden");
    dropZone.querySelector("i").classList.remove("hidden");
    dropZone.querySelector("p").classList.remove("hidden");
    dropZone.querySelector("small").classList.remove("hidden");
});

// 5. Analysis Logic
document.getElementById("analyze-btn").addEventListener("click", async () => {
    const textQuery = document.getElementById("text-query").value;
    const occasion = document.getElementById("query-occasion").value;
    const season = document.getElementById("query-season").value;

    if (!textQuery && !imageFile) {
        alert("Please provide an image or text description.");
        return;
    }

    const formData = new FormData();
    if (imageFile) formData.append("image", imageFile);
    formData.append("text_query", textQuery);
    formData.append("occasion", occasion);
    formData.append("season", season);

    // Show Modal Loading
    const responseModal = document.getElementById("response-modal");
    const adviceContent = document.getElementById("advice-content");
    const outfitResults = document.getElementById("outfit-results");
    
    responseModal.classList.remove("hidden");
    adviceContent.innerHTML = `<div class="loader"></div><p>Our AI is styling combinations for you...</p>`;
    outfitResults.innerHTML = '';

    try {
        const response = await fetch(`${API_BASE}/analyze-style`, {
            method: "POST",
            headers: { "Authorization": `Bearer ${authToken}` },
            body: formData
        });
        
        if (!response.ok) {
            adviceContent.innerHTML = `<p style="color:red">Synthesis failed.</p>`;
            return;
        }

        const data = await response.json();
        const advice = data.advice; // Now a dict {suggested_look, outfit_matches}
        const results = data.results;

        console.log("Full Advice Object (Analyze):", advice);
        

        // Render Advice (Structured JSON)
        if (advice) {
            adviceContent.innerHTML = `
                <div class="ai-text-response">
                    ${formatAIResponse(advice)}
                </div>
            `;
        } else {
            adviceContent.innerHTML = `<p>AI failed to generate styling advice.</p>`;
        }

        // Render Outfits
        outfitResults.innerHTML = '';
        if (results && results.length > 0) {
            results.forEach((item, index) => {
                const card = document.createElement("div");
                card.className = "outfit-card";
                card.innerHTML = `
                    <div style="position:relative;">
                        <img src="${API_BASE}/api/image/${item.id}" alt="${item.name}" loading="lazy" />
                    </div>
                    <h4>${item.name}</h4>
                    <p>${item.category} • ${item.price}</p>
                    <p class="score">Match: ${(item.similarity * 100).toFixed(1)}%</p>
                    <button class="like-btn" onclick='saveOutfit(${JSON.stringify(item)})'><i class="fa fa-heart"></i> Save to Collection</button>
                `;
                outfitResults.appendChild(card);
            });
        } else {
            outfitResults.innerHTML = "<p>No matching items found tailored to these specific criteria.</p>";
        }

    } catch (e) {
        adviceContent.innerHTML = `<p style="color:red">Synthesis failed: ${e.message}</p>`;
        console.error("Analysis Error:", e);
    }
});

// Close Modals
document.querySelectorAll(".close-modal").forEach(btn => {
    btn.addEventListener("click", () => {
        btn.closest(".modal").classList.add("hidden");
    });
});

// Save Outfit Logic
window.saveOutfit = async function(itemData) {
    try {
        const response = await fetch(`${API_BASE}/save-outfit`, {
            method: "POST",
            headers: { 
                "Content-Type": "application/json",
                "Authorization": `Bearer ${authToken}` 
            },
            body: JSON.stringify(itemData)
        });
        if(response.ok) {
            alert("Outfit saved!");
            updateStatsOnly(); // Refresh the counter
        }
    } catch(e) { console.error(e); }
}

async function updateStatsOnly() {
    try {
        const response = await fetch(`${API_BASE}/user-stats`, {
            headers: { "Authorization": `Bearer ${authToken}` }
        });
        if (response.ok) {
            const data = await response.json();
            document.getElementById("stat-saved").textContent = data.saved_count;
            document.getElementById("stat-recs").textContent = data.recommendation_count;
        }
    } catch (e) { console.error(e); }
}

// Profile Dropdown
const profileBtn = document.getElementById("profile-toggle");
const profileDropdown = document.getElementById("profile-dropdown");

if(profileBtn) {
    profileBtn.addEventListener("click", () => {
        profileDropdown.classList.toggle("hidden");
    });
}

// Close dropdown if clicked outside
document.addEventListener("click", (e) => {
    if(profileBtn && !profileBtn.contains(e.target) && !profileDropdown.contains(e.target)) {
        profileDropdown.classList.add("hidden");
    }
});

// Dropdown Links Logic (History and Saves go to same modal for UI simplicity here but fetch different ends)
document.getElementById("nav-saves").addEventListener("click", async (e) => {
    e.preventDefault();
    profileDropdown.classList.add("hidden");
    const modal = document.getElementById("saves-modal");
    const content = document.getElementById("saves-content");
    document.getElementById("saves-title").textContent = "Your Collection";
    modal.classList.remove("hidden");
    content.innerHTML = "Loading...";

    try {
        const response = await fetch(`${API_BASE}/saved-outfits`, {
            headers: { "Authorization": `Bearer ${authToken}` }
        });
        const data = await response.json();
        content.innerHTML = "";
        data.forEach(d => {
            content.innerHTML += `
                <div class="save-item">
                    <div class="save-img-wrapper">
                        <img src="${API_BASE}/api/image/${d.data.id}" loading="lazy" alt="${d.data.name}">
                    </div>
                    <div class="save-info">
                        <h4>${d.data.name}</h4>
                        <p>${d.data.price} - ${d.data.category}</p>
                    </div>
                    <div class="save-time">
                        <small>${new Date(d.time).toLocaleDateString()}</small>
                    </div>
                </div>
            `;
        });
    } catch(e) { content.innerHTML = `<p>Failed to load collection</p>`; }
});

document.getElementById("nav-history").addEventListener("click", async (e) => {
    e.preventDefault();
    profileDropdown.classList.add("hidden");
    const modal = document.getElementById("saves-modal");
    const content = document.getElementById("saves-content");
    document.getElementById("saves-title").textContent = "Recent Context History";
    modal.classList.remove("hidden");
    content.innerHTML = "Loading...";

    try {
        const response = await fetch(`${API_BASE}/history`, {
            headers: { "Authorization": `Bearer ${authToken}` }
        });
        const data = await response.json();
        content.innerHTML = "";
        data.forEach(d => {
            content.innerHTML += `
                <div class="history-item">
                    <h4>Prompt: "${d.prompt}"</h4>
                    <small>${new Date(d.time).toLocaleString()}</small>
                </div>
            `;
        });
    } catch(e) { content.innerHTML = `<p>Failed to load history</p>`; }
});

// Helper to format AI responses for high-fidelity display
function formatAIResponse(text) {
    if (!text) return "";

    // Split into Look 1 / Look 2 blocks if present
    // Matches "Look 1 —", "Look 2 —", "Look 1:", "Look 2:" etc.
    const lookSplit = text.split(/(?=Look\s[12]\s*[—–:-])/i);

    if (lookSplit.length > 1) {
        // Multi-look layout
        return lookSplit.map(block => {
            block = block.trim();
            if (!block) return "";

            // Extract the look title line (e.g. "Look 1 — The Riviera Brunch")
            const titleMatch = block.match(/^(Look\s[12]\s*[—–:-]\s*.+?)[\n]/i);
            let title = "";
            let body = block;

            if (titleMatch) {
                title = titleMatch[1].trim();
                body = block.slice(titleMatch[0].length).trim();
            } else {
                // Title might be whole first line
                const firstLine = block.indexOf("\n");
                if (firstLine !== -1) {
                    title = block.slice(0, firstLine).trim();
                    body = block.slice(firstLine).trim();
                }
            }
    // Split body into bullet points by sentence
            const points = splitToPoints(body);

            return `
                <div class="look-block">
                    <div class="look-title">${title}</div>
                    <ul class="look-points">
                        ${points.map(p => `<li>${p}</li>`).join("")}
                    </ul>
                </div>
            `;
        }).join("");
    }

    // Single response (follow-up question or short reply) — keep as plain text
    text = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/\n/g, "<br>");
    return `<p class="ai-plain">${text}</p>`;
}

// Splits a paragraph into individual sentences for bullet rendering
function splitToPoints(text) {
    if (!text) return [];

    // Clean up markdown artifacts
    text = text.replace(/\*\*(.*?)\*\*/g, "$1");
    text = text.replace(/\*(.*?)\*/g, "$1");
    text = text.replace(/^[-•]\s*/gm, "");

    // Split on sentence endings, keep the punctuation
    const raw = text.match(/[^.!?]+[.!?]+/g) || [text];

    return raw
        .map(s => s.trim())
        .filter(s => s.length > 10); // skip stray fragments
}

// Chat Logic
const openChatBtn = document.getElementById("open-chat-btn");
const chatPanel = document.getElementById("chat-panel");
const closeChat = document.getElementById("close-chat");
const chatMessages = document.getElementById("chat-messages");
const sendChatBtn = document.getElementById("send-chat");
const chatInput = document.getElementById("chat-input");

const uiOverlay = document.getElementById("ui-overlay");
const resizeChatBtn = document.getElementById("resize-chat");
const surpriseMeBtn = document.getElementById("surprise-me-btn");

openChatBtn.addEventListener("click", () => { 
    chatPanel.classList.remove("hidden"); 
    uiOverlay.classList.remove("hidden");
});
closeChat.addEventListener("click", () => { 
    chatPanel.classList.remove("expanded"); // Reset on close
    chatPanel.classList.add("hidden"); 
    uiOverlay.classList.add("hidden");
    resizeChatBtn.innerHTML = '<i class="fa fa-expand-alt"></i>';
});
resizeChatBtn.addEventListener("click", () => {
    chatPanel.classList.toggle("expanded");
    if(chatPanel.classList.contains("expanded")) {
        resizeChatBtn.innerHTML = '<i class="fa fa-compress-alt"></i>';
    } else {
        resizeChatBtn.innerHTML = '<i class="fa fa-expand-alt"></i>';
    }
});
surpriseMeBtn.addEventListener("click", () => {
    chatInput.value = "🎲 Surprise me with a quick suggestion!";
    sendMessage();
});
uiOverlay.addEventListener("click", () => {
    chatPanel.classList.add("hidden"); 
    uiOverlay.classList.add("hidden");
});

// Quick suggestions logic
document.querySelectorAll(".sug-btn").forEach(btn => {
    btn.addEventListener("click", () => {
        chatInput.value = btn.textContent;
        sendMessage();
    });
});

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    // Add user message
    chatMessages.innerHTML += `<div class="message user">${text}</div>`;
    chatInput.value = "";
    chatMessages.scrollTop = chatMessages.scrollHeight;

    // Add typing indicator
    const typingId = "typing-" + Date.now();
    chatMessages.innerHTML += `
        <div class="typing-indicator" id="${typingId}">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    `;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    try {
    const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { 
            "Content-Type": "application/json",
            "Authorization": `Bearer ${authToken}` 
        },
        body: JSON.stringify({ message: text })
    });

    const data = await response.json();
    document.getElementById(typingId).remove();

    if (!response.ok) {
        throw new Error(data.detail || "Synthesis failed");
    }

    const aiText = data.response || "Styling complete.";
     chatMessages.innerHTML += `
        <div class="message ai">
            ${formatAIResponse(aiText)}
        </div>
    `;

    chatMessages.scrollTop = chatMessages.scrollHeight;

} catch (e) {
    if (document.getElementById(typingId)) {
        document.getElementById(typingId).remove();
    }
    console.error("Chat Error:", e);
    chatMessages.innerHTML += `<div class="message ai text-red">⚠️ Error: ${e.message}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
}
sendChatBtn.addEventListener("click", sendMessage);
chatInput.addEventListener("keyup", (e) => { if (e.key === 'Enter') sendMessage(); });
