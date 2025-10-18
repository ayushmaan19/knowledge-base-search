import "dotenv/config";
import express from "express";
import cors from "cors";
import multer from "multer";
import pdf from "pdf-parse/lib/pdf-parse.js";
import fs from "fs/promises";
import Groq from "groq-sdk";
import { HfInference } from "@huggingface/inference";
import { pipeline } from "@xenova/transformers";
import { GoogleGenerativeAI } from "@google/generative-ai";

const app = express();
const port = process.env.PORT || 5001;
app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });
// const INDEX_PATH = "./vector_store.faiss"; // FAISS removed
const DOCS_PATH = "./documents.json";
const EMBEDDINGS_PATH = "./embeddings.json";
app.post("/clear", async (req, res) => {
  try {
    await fs.writeFile(DOCS_PATH, JSON.stringify([]));
    await fs.writeFile(EMBEDDINGS_PATH, JSON.stringify([]));

    const deleteCandidates = [
      "http://localhost:8000/collections/knowledge-base",
      "http://localhost:8000/api/v1/collections/knowledge-base",
    ];
    for (const url of deleteCandidates) {
      try {
        await fetch(url, { method: "DELETE" });
      } catch (_) {
        // ignore and try next
      }
    }

    res.json({ success: true, message: "Knowledge base cleared." });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.get("/models", async (req, res) => {
  const groqModels = await fetchGroqModelsSafe();
  res.json({
    groq: {
      enabled: !!process.env.GROQ_API_KEY,
      available: groqModels,
      selected:
        process.env.GROQ_MODEL ||
        (await pickGroqModel(["llama3", "mixtral", "gemma", "qwen"])),
    },
    gemini: {
      enabled: !!genAI,
      selected: process.env.GEMINI_MODEL || "gemini-2.0-flash",
    },
  });
});
// Initialize Groq and Hugging Face clients (no LangChain)
let groq = null;
if (process.env.GROQ_API_KEY) {
  groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
}
const hfToken =
  process.env.HUGGINGFACE_API_KEY || process.env.HUGGINGFACEHUB_API_KEY;
const hf = hfToken ? new HfInference(hfToken) : null;
let localEmbedder = null; // lazy-init Xenova pipeline
// Gemini client (optional)
const geminiApiKey = process.env.GEMINI_API_KEY;
const genAI = geminiApiKey ? new GoogleGenerativeAI(geminiApiKey) : null;

// ---- Groq model discovery and fallback ----
let cachedGroqModels = null; 
async function fetchGroqModelsSafe() {
  if (!process.env.GROQ_API_KEY) return [];
  try {
    const resp = await fetch("https://api.groq.com/openai/v1/models", {
      headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` },
    });
    if (!resp.ok) return [];
    const json = await resp.json();
    // API returns { data: [{id: '...'}, ...] }
    const ids = Array.isArray(json?.data)
      ? json.data.map((m) => m.id).filter(Boolean)
      : Array.isArray(json)
      ? json.map((m) => m.id || m)
      : [];
    cachedGroqModels = ids;
    return ids;
  } catch (_) {
    return [];
  }
}

async function pickGroqModel(preferredOrder = []) {
  const list = cachedGroqModels || (await fetchGroqModelsSafe());
  if (list && list.length) {
    // Try preferred names in order
    for (const name of preferredOrder) {
      const match = list.find((id) => id.includes(name));
      if (match) return match;
    }
    const llama = list.find((id) => id.toLowerCase().includes("llama"));
    if (llama) return llama;
    const mixtral = list.find((id) => id.toLowerCase().includes("mixtral"));
    if (mixtral) return mixtral;
    return list[0];
  }
  // Fallback if API not reachable: a commonly available model name
  return "mixtral-8x7b-32768";
}

async function generateWithGroq({ system, user }) {
  let model =
    process.env.GROQ_MODEL ||
    (await pickGroqModel(["llama3", "mixtral", "gemma", "qwen"]));
  try {
    const completion = await groq.chat.completions.create({
      model,
      temperature: 0.2,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    });
    const text = completion.choices?.[0]?.message?.content?.trim() || "";
    return { text, model };
  } catch (err) {
    const msg = err?.message || "";
    if (msg.includes("decommissioned") || msg.toLowerCase().includes("model")) {
      const alt = await pickGroqModel(["llama-3", "mixtral", "gemma", "qwen"]);
      const completion = await groq.chat.completions.create({
        model: alt,
        temperature: 0.2,
        messages: [
          { role: "system", content: system },
          { role: "user", content: user },
        ],
      });
      const text = completion.choices?.[0]?.message?.content?.trim() || "";
      return { text, model: alt };
    }
    throw err;
  }
}

async function generateWithGemini({ system, user }) {
  if (!genAI) throw new Error("Gemini not configured");
  const modelId = process.env.GEMINI_MODEL || "gemini-1.5-flash";
  const model = genAI.getGenerativeModel({ model: modelId });
  const prompt = `${system}\n\n${user}`;
  const result = await model.generateContent([{ text: prompt }]);
  const text = result?.response?.text?.() || "";
  return { text, model: modelId };
}

async function generateAnswerWithFallback({ system, user }) {
  if (groq) {
    try {
      const out = await generateWithGroq({ system, user });
      if (out.text) return { ...out, provider: "groq" };
    } catch (e) {
      console.warn("Groq generation failed, will try Gemini:", e?.message || e);
    }
  }
  if (genAI) {
    const out = await generateWithGemini({ system, user });
    if (out.text) return { ...out, provider: "gemini" };
  }
  throw new Error(
    "No provider produced a response. Configure GROQ_API_KEY or GEMINI_API_KEY."
  );
}

async function embedTexts(texts) {
  const modelId =
    process.env.HF_EMBEDDING_MODEL || "sentence-transformers/all-MiniLM-L6-v2";
  if (hf) {
    const out = await Promise.all(
      texts.map(async (t) => {
        const res = await hf.featureExtraction({ model: modelId, inputs: t });
        const vecLike = Array.isArray(res?.[0]) ? res[0] : res;
        return Array.from(vecLike || []);
      })
    );
    return out;
  }
  // Fallback to local embeddings (no token needed)
  if (!localEmbedder) {
    // Xenova model uses its own naming; map common sentence-transformers to Xenova equivalents
    const xenovaModel =
      process.env.XENOVA_EMBEDDING_MODEL || "Xenova/all-MiniLM-L6-v2";
    localEmbedder = await pipeline("feature-extraction", xenovaModel, {
      quantized: true,
    });
  }
  const outputs = [];
  for (const t of texts) {
    const res = await localEmbedder(t, { pooling: "mean", normalize: true });
    outputs.push(Array.from(res.data));
  }
  return outputs;
}

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// ========================= 📄 MULTI-PDF UPLOAD ROUTE =========================
app.post("/upload", upload.array("files"), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: "No files provided." });
  }
  try {
    // Load previous documents and embeddings if they exist
    let prevChunks = [];
    let prevEmbeddings = [];
    try {
      const prevData = await fs.readFile(DOCS_PATH, "utf-8");
      prevChunks = JSON.parse(prevData);
    } catch (e) {}
    try {
      const prevEmb = await fs.readFile(EMBEDDINGS_PATH, "utf-8");
      prevEmbeddings = JSON.parse(prevEmb);
    } catch (e) {}
    let allChunks = [...prevChunks];
    let allEmbeddings = [...prevEmbeddings];
    let fileNames = [];
    for (const file of req.files) {
      console.log(`1. Parsing PDF: ${file.originalname}`);
      const pdfData = await pdf(file.buffer);
      const text = pdfData.text;
      console.log(`2. Chunking text for ${file.originalname}...`);
      const chunks = [];
      for (let i = 0; i < text.length; i += 800) {
        chunks.push(text.substring(i, i + 1000));
      }
      if (chunks.length === 0) continue;
      console.log(
        `3. Creating embeddings for ${chunks.length} chunks in ${file.originalname}...`
      );
      const embeddings = await embedTexts(chunks);
      if (!embeddings?.length) continue;
      allChunks.push(...chunks);
      allEmbeddings.push(...embeddings);
      fileNames.push(file.originalname);
    }
    if (!allChunks.length || !allEmbeddings.length) {
      return res
        .status(400)
        .json({ error: "No valid PDFs or embeddings created." });
    }
    // Deduplicate and filter out invalid embeddings
    const seen = new Set();
    const dedupChunks = [];
    const dedupEmbeddings = [];
    // Iterate from end so the most recently added chunk (with correct file metadata) wins
    for (let i = allChunks.length - 1; i >= 0; i--) {
      const chunk = allChunks[i];
      const emb = allEmbeddings[i];
      if (
        !seen.has(chunk) &&
        Array.isArray(emb) &&
        emb.length > 0 &&
        emb.every((v) => typeof v === "number" && Number.isFinite(v))
      ) {
        seen.add(chunk);
        // unshift to preserve original order while keeping latest metadata
        dedupChunks.unshift(chunk);
        dedupEmbeddings.unshift(emb);
      }
    }
    // Debug: print embeddings shape
    console.log(
      "Total Chunks:",
      dedupChunks.length,
      "Total Embeddings:",
      dedupEmbeddings.length
    );
    // --- ChromaDB integration ---
    try {
      const response = await fetch(
        "http://localhost:8000/collections/knowledge-base/documents",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            documents: dedupChunks,
            embeddings: dedupEmbeddings,
          }),
        }
      );
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`ChromaDB upload failed: ${text}`);
      }
      await fs.writeFile(DOCS_PATH, JSON.stringify(dedupChunks));
      await fs.writeFile(EMBEDDINGS_PATH, JSON.stringify(dedupEmbeddings));
      console.log("✅ Upload complete! (ChromaDB)");
      res.json({
        message: `${fileNames.join(", ")} processed and sent to ChromaDB.`,
      });
    } catch (error) {
      console.error("❌ Error uploading to ChromaDB:", error);
      res.status(500).json({
        error: "Failed to upload to ChromaDB",
        message: error.message,
      });
    }
  } catch (error) {
    console.error("❌ Error in /upload:", error);
    const status = error?.response?.status || 500;
    const details = error?.response?.data || error?.message;
    res
      .status(status)
      .json({ error: "Failed to process files.", message: details });
  }
});

// ========================= 💬 QUERY ROUTE =========================
app.post("/query", async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).json({ error: "Query missing." });

  try {
    console.log("1. Creating query embedding...");
    const [queryEmbedding] = await embedTexts([query]);
    if (!queryEmbedding.length || !Number.isFinite(queryEmbedding[0])) {
      return res.status(502).json({
        error: "Embedding service returned an unexpected format for query.",
      });
    }

    console.log("2. Querying ChromaDB for relevant context...");
    // Query ChromaDB for top 3 similar documents
    let context = "";
    try {
      const response = await fetch(
        "http://localhost:8000/collections/knowledge-base/query",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            embedding: queryEmbedding,
            top_k: 3,
          }),
        }
      );
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`ChromaDB query failed: ${text}`);
      }
      const result = await response.json();
      context = result.documents?.join("\n---\n") || "";
    } catch (error) {
      console.error("❌ Error querying ChromaDB:", error);
      return res
        .status(500)
        .json({ error: "Failed to query ChromaDB", message: error.message });
    }

    // If no relevant context was found, return a clear, user-friendly fallback
    if (!context || !context.trim()) {
      return res.json({
        answer: "This question is not relevant to the uploaded documents.",
        provider: "none",
        model: "none",
      });
    }

    // If no provider configured, degrade gracefully and return relevant excerpts
    if (!groq && !genAI) {
      const excerpts =
        context.length > 1200 ? context.slice(0, 1200) + "\n..." : context;
      return res.json({
        answer: `Relevant excerpts from your documents:\n\n${excerpts}`,
        provider: "none",
        model: "none",
      });
    }

    console.log("3. Generating answer...");
    const system =
      "You are a helpful assistant. Answer succinctly using ONLY the provided context. If the context does not contain the answer, reply exactly: 'This question is not relevant to the uploaded documents.'";
    const user = `Context:\n${context}\n\nQuestion: ${query}\n\nAnswer:`;
    const {
      text: answer,
      provider,
      model,
    } = await generateAnswerWithFallback({ system, user });
    res.json({ answer, provider, model });
  } catch (error) {
    console.error("❌ Error in /query:", error);
    res.status(500).json({ error: "Failed to get answer." });
  }
});

// List available Groq models
app.get("/groq-models", async (req, res) => {
  try {
    if (!process.env.GROQ_API_KEY)
      return res.status(400).json({ error: "GROQ_API_KEY not set" });
    const resp = await fetch("https://api.groq.com/openai/v1/models", {
      headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` },
    });
    if (!resp.ok) {
      const text = await resp.text();
      return res
        .status(resp.status)
        .json({ error: "Failed to list Groq models", details: text });
    }
    const json = await resp.json();
    return res.json(json);
  } catch (err) {
    console.error("Error listing Groq models:", err);
    return res
      .status(500)
      .json({ error: "Error listing Groq models", message: err?.message });
  }
});

// Health check endpoint showing backend configuration
app.get("/health", async (req, res) => {
  const embeddingBackend = hf ? "HuggingFace API" : "Xenova (local)";
  const embeddingModel = hf
    ? process.env.HF_EMBEDDING_MODEL || "sentence-transformers/all-MiniLM-L6-v2"
    : process.env.XENOVA_EMBEDDING_MODEL || "Xenova/all-MiniLM-L6-v2";
  const groqModel =
    process.env.GROQ_MODEL ||
    (await pickGroqModel(["llama3", "mixtral", "gemma", "qwen"]));
  const availableModels = await fetchGroqModelsSafe();
  const status = {
    status: "ok",
    port,
    embedding: {
      backend: embeddingBackend,
      model: embeddingModel,
      apiKeySet: !!hfToken,
    },
    groq: {
      model: groqModel,
      apiKeySet: !!process.env.GROQ_API_KEY,
      availableCount: availableModels.length,
    },
    gemini: {
      configured: !!genAI,
      model: process.env.GEMINI_MODEL || "gemini-1.5-flash",
      apiKeySet: !!geminiApiKey,
    },
    chromadb: {
      url: "http://localhost:8000",
      docsPath: DOCS_PATH,
    },
  };
  res.json(status);
});

app.listen(port, () => {
  console.log(`✅ Backend running at: http://localhost:${port}`);
});
