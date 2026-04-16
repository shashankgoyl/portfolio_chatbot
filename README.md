# 🤖 Portfolio AI Chatbot — Shashank Goyal

A **fully free** RAG-powered chatbot for your portfolio website.  
Built with **Groq LLM + FAISS vector DB + FastAPI**, deployable to **Northflank** (free tier).

---

## 📁 Project Structure

```
portfolio-chatbot/
├── main.py              # FastAPI server + conversation management
├── rag_system.py        # FAISS RAG engine + Groq LLM integration
├── config.py            # All configuration (edit owner info here)
├── resume.txt           # ← YOUR RESUME DATA (edit this!)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container image for deployment
├── docker-compose.yml   # For local Docker testing
├── .env.example         # Environment variable template
├── static/
│   └── index.html       # Frontend chat UI
└── README.md
```

---

## ⚡ Quick Start (Local — No Docker)

### Step 1 — Prerequisites
- Python 3.11+
- Git

### Step 2 — Clone / Download & enter folder
```bash
cd portfolio-chatbot
```

### Step 3 — Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```
> ⚠️ First install takes 3–5 min — downloads sentence-transformers model (~80 MB)

### Step 5 — Set up environment variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your Groq API key
# Get a FREE key at: https://console.groq.com
```

Open `.env` and replace `your_groq_api_key_here` with your actual key:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
```

### Step 6 — Update your resume data
Open `resume.txt` and update it with your own information.  
The more detail you add, the better the chatbot answers!

### Step 7 — Run the server
```bash
python main.py
```

### Step 8 — Open the chatbot
- **Chat UI:** http://localhost:8000/ui
- **API Docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

> 💡 Wait ~30–60 seconds for the FAISS index to build on first startup.

---

## 🐳 Local Docker Testing

### Step 1 — Build and run
```bash
# Make sure .env file exists with your GROQ_API_KEY
docker-compose up --build
```

### Step 2 — Open
- http://localhost:8000/ui

---

## ☁️ Deploy to Northflank (Free Tier)

### Prerequisites
- GitHub account
- Northflank account (free at https://northflank.com)

### Step 1 — Push to GitHub
```bash
# Create a new repo on GitHub first, then:
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/portfolio-chatbot.git
git push -u origin main
```

### Step 2 — Create Northflank account
1. Go to https://northflank.com and sign up (free)
2. Create a new **Project**

### Step 3 — Create a Deployment Service
1. Inside your project, click **"New Service"**
2. Choose **"Deployment"**
3. Connect your **GitHub repository**
4. Northflank auto-detects your `Dockerfile` ✅

### Step 4 — Set Environment Variables
In the service settings, go to **"Environment"** and add:
```
GROQ_API_KEY = gsk_xxxxxxxxxxxxxxxxxxxx
PORT         = 8000
```

### Step 5 — Set Port
In **"Networking"** settings:
- Container port: `8000`
- Enable public URL ✅

### Step 6 — Deploy!
Click **"Deploy"** — Northflank will:
1. Pull your GitHub repo
2. Build the Docker image (5–10 min on first build)
3. Start the container

### Step 7 — Get your public URL
Northflank gives you a URL like:  
`https://your-service-name.northflank.app`

### Step 8 — Update the frontend
Open `static/index.html` and update line:
```javascript
const API_BASE = "https://your-service-name.northflank.app";
```

Then push and redeploy, OR host the HTML separately on GitHub Pages / Netlify.

---

## 🌐 Using the HTML Frontend on Your Portfolio

### Option A — Serve from FastAPI (already set up)
Visit: `https://your-app.northflank.app/ui`

### Option B — Host on Netlify/GitHub Pages
1. Update `API_BASE` in `static/index.html` to your Northflank URL
2. Upload `static/index.html` to Netlify / GitHub Pages
3. It will connect to your Northflank backend via CORS

### Option C — Embed in existing site
Add an iframe:
```html
<iframe 
  src="https://your-app.northflank.app/ui" 
  width="400" height="600" 
  style="border:none; border-radius:16px;"
></iframe>
```

---

## 🔧 API Reference

### POST /chat
```json
{
  "message": "What are Shashank's skills?",
  "conversation_id": null   // omit for new conversation
}
```
Response:
```json
{
  "answer": "Shashank is skilled in Python, LangChain, FastAPI...",
  "conversation_id": "uuid-here",
  "sources": ["resume.txt"],
  "mode": "rag",
  "processing_time_ms": 820
}
```

### GET /health
Returns system status, chunk count, RAG mode.

### GET /stats
Returns active conversations and turn counts.

### DELETE /conversation/{id}
Clears conversation history.

---

## 🎨 Customising

### Change bot name / info
Edit `config.py`:
```python
OWNER_NAME = "Your Name"
OWNER_EMAIL = "you@example.com"
OWNER_GITHUB = "github.com/yourusername"
```

### Update resume data
Edit `resume.txt` — add projects, blog posts, achievements, anything!

### Change LLM model
In `.env`:
```
GROQ_MODEL=llama-3.3-70b-versatile   # More capable but slower
GROQ_MODEL=llama-3.1-8b-instant      # Fast (default)
GROQ_MODEL=gemma2-9b-it              # Alternative
```

---

## 🆓 Free Tier Limits

| Service | Free Limit |
|---------|-----------|
| Groq API | ~14,400 req/day on free tier |
| Northflank | 1 service, 0.5 vCPU, 512 MB RAM |
| sentence-transformers | Unlimited (runs locally) |
| FAISS | Unlimited (runs locally) |

---

## 🐛 Troubleshooting

**"Warming up" message persists > 2 min**
→ Check logs in Northflank dashboard for errors
→ Verify `GROQ_API_KEY` is set correctly

**ModuleNotFoundError**
→ Run `pip install -r requirements.txt` again

**CORS errors in browser**
→ Check `API_BASE` URL in `index.html` matches your deployment URL

**Out of memory on Northflank**
→ Free tier has 512 MB. The model needs ~200 MB. If it crashes, upgrade to Northflank Sandbox ($0 with credits).

---

## 🚀 Future Ideas (React Frontend)

When you build your React frontend:
1. Replace `API_BASE` with your Northflank URL
2. Use `fetch('/chat', {...})` with the same request/response schema
3. Store `conversation_id` in React state for multi-turn chat
4. Add streaming responses using SSE (contact me for help!)

---

Built with ❤️ by Shashank Goyal  
shashankgoyal902@gmail.com | github.com/shashankgoyl
