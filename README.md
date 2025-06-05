```markdown
# Foodhak Barcode Recommendation AI Service

Real-time, AI-powered nutrition verdicts based on barcode scans and user profiles. Get concise, personalized food recommendations via WebSocket—instantly.

---

## 🌍 Environments

- **Production API:** `https://ai-foodhak.com`
- **Staging API:** `https://staging.ai-foodhak.com`

---

## 🚦 How It Works

1. **POST to `/barcode-recommend`**  
   Supply your `user_id` and the barcode.  
   Receive a WebSocket URL to get real-time recommendations.

2. **Connect to WebSocket URL**  
   Send the barcode as JSON.  
   Receive a streamed, AI-generated nutrition summary—tailored to the user's health profile, goals, restrictions, and the scanned product.

3. **Powered by Anthropic Claude 3 (primary)**  
   - **OpenAI fallback:** Automatic if Claude is overloaded.

---

## 🔑 Authentication

All endpoints require:  
```

Authorization: Bearer \<API\_KEY>

````

---

## 🛠️ Usage

### 1. Get a Recommendation WebSocket URL

#### Production

```bash
curl -X POST https://ai-foodhak.com/barcode-recommend \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "USER123", "barcode": "3029330003533"}'
````


**Response:**

```json
{
  "websocket_url": "wss://ai-foodhak.com/ws/recommend/USER123",
  "message": "Please use this WebSocket URL to connect for real-time recommendations."
}
```

---

### 2. Connect to WebSocket for Real-time Recommendation

**WebSocket Endpoint:**

* Production: `wss://ai-foodhak.com/ws/recommend/{user_id}`
* Staging: `wss://staging.ai-foodhak.com/ws/recommend/{user_id}`

**Send (as JSON):**

```json
{
  "barcode": "3029330003533"
}
```

**Receive (streamed JSON):**

* `type: streaming` (the AI-generated response in chunks)
* `type: message_stop` (completion signal)
* `type: error` (for any errors)

---

#### Example: Python WebSocket Client

```python
import websockets
import asyncio
import json

async def get_recommendation(user_id, barcode):
    uri = f"wss://ai-foodhak.com/ws/recommend/{user_id}"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"barcode": barcode}))
        async for message in ws:
            print(message)

asyncio.run(get_recommendation("USER123", "3029330003533"))
```

---

### 3. Health Check

Check service status and health:

```bash
curl https://ai-foodhak.com/health
```

**Response:**

```json
{
  "status": "healthy",
  "message": "Service is up and running."
}
```

---

## 📦 API Reference

| Method | Endpoint                  | Description                      |
| ------ | ------------------------- | -------------------------------- |
| POST   | `/barcode-recommend`      | Get recommendation WebSocket URL |
| WS     | `/ws/recommend/{user_id}` | Real-time nutrition verdict      |
| GET    | `/health`                 | Service health check             |

---

## ⚡ Features

* **AI-powered:** Uses Claude 3 and OpenAI for personalized nutrition verdicts.
* **Seamless fallback:** Fails over from Claude to OpenAI if service is overloaded.
* **Context-rich:** Factors in user profile, goals, allergens, and full ingredient/nutrient analysis.
* **UI-friendly:** 2–4 sentence, Markdown-formatted responses—ready for mobile display.
* **Secure:** Requires API key for all calls.

---

## 🔒 Security

* Never expose your API Key.
* All requests must include a valid Bearer token.

---

## 📝 Notes

* Missing data and errors are transparently handled and communicated in responses.
* Rate-limiting and fallback handled automatically.

---