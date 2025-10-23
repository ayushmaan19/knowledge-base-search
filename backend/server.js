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
      } catch (_) {}
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

let groq = null;
if (process.env.GROQ_API_KEY) {
  groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
}
const hfToken =
  process.env.HUGGINGFACE_API_KEY || process.env.HUGGINGFACEHUB_API_KEY;
const hf = hfToken ? new HfInference(hfToken) : null;
let localEmbedder = null;
let localEmbedderReady = null;
const EMBEDDING_CACHE_MAX = 500;
const embeddingCache = new Map();
const geminiApiKey = process.env.GEMINI_API_KEY;
const genAI = geminiApiKey ? new GoogleGenerativeAI(geminiApiKey) : null;

let cachedGroqModels = null;
async function fetchGroqModelsSafe() {
  if (!process.env.GROQ_API_KEY) return [];
  try {
    const resp = await fetch("https://api.groq.com/openai/v1/models", {
      headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` },
    });
    if (!resp.ok) return [];
    const json = await resp.json();
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
  const modelId = process.env.GEMINI_MODEL || "gemini-2.0-flash";
  const model = genAI.getGenerativeModel({ model: modelId });
  const prompt = `${system}\n\n${user}`;
  const result = await model.generateContent([{ text: prompt }]);
  const text = result?.response?.text?.() || "";
  return { text, model: modelId };
}

async function generateAnswerWithFallback({ system, user }) {
  const GROQ_TIMEOUT_MS = parseInt(process.env.GROQ_TIMEOUT_MS || "10000", 10);
  const GEMINI_TIMEOUT_MS = parseInt(
    process.env.GEMINI_TIMEOUT_MS || "15000",
    10
  );

  if (groq) {
    try {
      const groqPromise = generateWithGroq({ system, user });
      const out = await Promise.race([
        groqPromise,
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("Groq generation timeout")),
            GROQ_TIMEOUT_MS
          )
        ),
      ]);
      if (out?.text) return { ...out, provider: "groq" };
    } catch (e) {
      console.warn(
        "Groq generation failed/timeout, will try Gemini:",
        e?.message || e
      );
    }
  }

  if (genAI) {
    try {
      const gemPromise = generateWithGemini({ system, user });
      const out = await Promise.race([
        gemPromise,
        new Promise((_, reject) =>
          setTimeout(
            () => reject(new Error("Gemini generation timeout")),
            GEMINI_TIMEOUT_MS
          )
        ),
      ]);
      if (out?.text) return { ...out, provider: "gemini" };
    } catch (e) {
      console.warn("Gemini generation failed/timeout:", e?.message || e);
    }
  }
  throw new Error(
    "No provider produced a response. Configure GROQ_API_KEY or GEMINI_API_KEY."
  );
}

async function embedTexts(texts) {
  const modelId =
    process.env.HF_EMBEDDING_MODEL || "sentence-transformers/all-MiniLM-L6-v2";
  const start = Date.now();
  if (hf) {
    const HF_TIMEOUT_MS = parseInt(process.env.HF_TIMEOUT_MS || "5000", 10);
    const calls = texts.map(async (t) => {
      try {
        const hfPromise = hf.featureExtraction({ model: modelId, inputs: t });
        const res = await Promise.race([
          hfPromise,
          new Promise((_, reject) =>
            setTimeout(
              () => reject(new Error("HF embed timeout")),
              HF_TIMEOUT_MS
            )
          ),
        ]);
        const vecLike = Array.isArray(res?.[0]) ? res[0] : res;
        const vec = Array.from(vecLike || []);
        const key = t.slice(0, 200);
        embeddingCache.set(key, vec);
        if (embeddingCache.size > EMBEDDING_CACHE_MAX) {
          const firstKey = embeddingCache.keys().next().value;
          embeddingCache.delete(firstKey);
        }
        return vec;
      } catch (e) {
        console.warn(
          "HF embedding failed or timed out, falling back to local:",
          e?.message || e
        );
      }
      if (!localEmbedderReady) {
        localEmbedderReady = (async () => {
          const xenovaModel =
            process.env.XENOVA_EMBEDDING_MODEL || "Xenova/all-MiniLM-L6-v2";
          localEmbedder = await pipeline("feature-extraction", xenovaModel, {
            quantized: true,
          });
          return true;
        })();
      }
      await localEmbedderReady;
      if (embeddingCache.has(t.slice(0, 200))) {
        const val = embeddingCache.get(t.slice(0, 200));
        embeddingCache.delete(t.slice(0, 200));
        embeddingCache.set(t.slice(0, 200), val);
        return val;
      }
      const resLocal = await localEmbedder(t, {
        pooling: "mean",
        normalize: true,
      });
      const vecLocal = Array.from(resLocal.data || []);
      embeddingCache.set(t.slice(0, 200), vecLocal);
      if (embeddingCache.size > EMBEDDING_CACHE_MAX) {
        const firstKey = embeddingCache.keys().next().value;
        embeddingCache.delete(firstKey);
      }
      return vecLocal;
    });
    const out = await Promise.all(calls);
    console.log(
      `ℹ️ embedTexts (HF path) processed ${texts.length} texts in ${
        Date.now() - start
      }ms`
    );
    return out;
  }
  if (!localEmbedderReady) {
    // Kick off initialization now so we don't block for long on first request.
    localEmbedderReady = (async () => {
      const xenovaModel =
        process.env.XENOVA_EMBEDDING_MODEL || "Xenova/all-MiniLM-L6-v2";
      localEmbedder = await pipeline("feature-extraction", xenovaModel, {
        quantized: true,
      });
      return true;
    })();
  }

  await localEmbedderReady;

  const tasks = texts.map(async (t) => {
    const key = t.slice(0, 200);
    if (embeddingCache.has(key)) {
      const val = embeddingCache.get(key);
      embeddingCache.delete(key);
      embeddingCache.set(key, val);
      return val;
    }
    const res = await localEmbedder(t, { pooling: "mean", normalize: true });
    const vec = Array.from(res.data || []);
    embeddingCache.set(key, vec);
    if (embeddingCache.size > EMBEDDING_CACHE_MAX) {
      const firstKey = embeddingCache.keys().next().value;
      embeddingCache.delete(firstKey);
    }
    return vec;
  });
  const outputs = await Promise.all(tasks);
  return outputs;
}

app.post("/upload", upload.array("files"), async (req, res) => {
  if (!req.files || req.files.length === 0) {
    return res.status(400).json({ error: "No files provided." });
  }
  try {
    let prevChunks = [];
    let prevEmbeddings = [];
    let prevMetadatas = [];
    try {
      const prevData = await fs.readFile(DOCS_PATH, "utf-8");
      prevChunks = JSON.parse(prevData);
    } catch (e) {}
    try {
      const prevEmb = await fs.readFile(EMBEDDINGS_PATH, "utf-8");
      prevEmbeddings = JSON.parse(prevEmb);
    } catch (e) {}
    prevMetadatas = prevChunks.map(() => ({ file: "unknown" }));

    let allChunks = [...prevChunks];
    let allEmbeddings = [...prevEmbeddings];
    let allMetadatas = [...prevMetadatas];
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
      // add metadata for each chunk with the file name
      for (let j = 0; j < chunks.length; j++) {
        allMetadatas.push({ file: file.originalname });
      }
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
    const dedupMetadatas = [];
    // Iterate from end so the most recently added chunk (with correct file metadata) wins
    for (let i = allChunks.length - 1; i >= 0; i--) {
      const chunk = allChunks[i];
      const emb = allEmbeddings[i];
      const meta = allMetadatas[i];
      if (
        !seen.has(chunk) &&
        Array.isArray(emb) &&
        emb.length > 0 &&
        emb.every((v) => typeof v === "number" && Number.isFinite(v))
      ) {
        seen.add(chunk);
        dedupChunks.unshift(chunk);
        dedupEmbeddings.unshift(emb);
        dedupMetadatas.unshift(meta);
      }
    }
    console.log(
      "Total Chunks:",
      dedupChunks.length,
      "Total Embeddings:",
      dedupEmbeddings.length
    );
    try {
      const response = await fetch(
        "http://localhost:8000/collections/knowledge-base/documents",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            documents: dedupChunks,
            embeddings: dedupEmbeddings,
            metadatas: dedupMetadatas,
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

app.post("/query", async (req, res) => {
  const { query, file } = req.body;
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
    let context = "";
    try {
      const body = {
        embedding: queryEmbedding,
        top_k: 3,
        include: ["documents", "metadatas", "distances", "ids"],
      };
      if (file && file !== "all") {
        body.where = { file: { $eq: file } };
      }

      const CHROMA_TIMEOUT_MS = parseInt(
        process.env.CHROMA_TIMEOUT_MS || "5000",
        10
      );
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), CHROMA_TIMEOUT_MS);
      const chromaUrl =
        process.env.CHROMA_URL ||
        "http://127.0.0.1:8000/collections/knowledge-base/query";
      const fetchStart = Date.now();
      let response;
      try {
        response = await fetch(chromaUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timeout);
        console.log(`ℹ️ ChromaDB fetch took ${Date.now() - fetchStart}ms`);
      }
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`ChromaDB query failed: ${text}`);
      }
      const result = await response.json();
      let documents = Array.isArray(result?.documents) ? result.documents : [];
      const metadatas = Array.isArray(result?.metadatas)
        ? result.metadatas
        : [];
      if (file && file !== "all" && metadatas.length) {
        const filtered = documents.filter(
          (d, i) => metadatas[i]?.file === file
        );
        documents = filtered;
      }
      context = documents.join("\n---\n") || "";
    } catch (error) {
      console.error("❌ Error querying ChromaDB:", error);
      return res
        .status(500)
        .json({ error: "Failed to query ChromaDB", message: error.message });
    }
    if (!context || !context.trim()) {
      return res.json({
        answer: "This question is not relevant to the uploaded documents.",
        provider: "none",
        model: "none",
      });
    }

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

app.get("/groq-models", async (req, res) => {
  try {
    if (!process.env.GROQ_API_KEY)
      return res.status(400).json({ error: "GROQ_API_KEY not set" });
    const resp = await fetch("https://api.groq.com/openai/v1/models", {
      headers: { Authorization: `Bearer ${process.env.GROQ_API_KEY}` },
    });
    if (!resp.ok) {
      const text = await response.text();
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

if (!hf) {
  if (!localEmbedderReady) {
    console.log(
      "ℹ️ No HuggingFace API key set - warming local Xenova embedder in background..."
    );
    localEmbedderReady = (async () => {
      try {
        const xenovaModel =
          process.env.XENOVA_EMBEDDING_MODEL || "Xenova/all-MiniLM-L6-v2";
        localEmbedder = await pipeline("feature-extraction", xenovaModel, {
          quantized: true,
        });
        console.log("✅ Local Xenova embedder ready.");
        return true;
      } catch (e) {
        console.error(
          "⚠️ Failed to initialize local Xenova embedder:",
          e?.message || e
        );
        localEmbedderReady = null;
        throw e;
      }
    })();
  }
}
