import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import multer from 'multer';
import fs from 'fs/promises';
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf"; // <-- This is the corrected line
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const app = express();
const port = 5001;
app.use(cors());
app.use(express.json());

const upload = multer({ storage: multer.memoryStorage() });
const VECTOR_STORE_PATH = "./vector_store";

app.post('/upload', async (req, res) => {
    if (!req.file) return res.status(400).json({ error: "No file uploaded." });

    try {
        await fs.rm(VECTOR_STORE_PATH, { recursive: true, force: true });
        const pdfLoader = new PDFLoader(new Blob([req.file.buffer]));
        const docs = await pdfLoader.load();
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
        const splitDocs = await textSplitter.splitDocuments(docs);

        const embeddings = new HuggingFaceInferenceEmbeddings();

        const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);
        await vectorStore.save(VECTOR_STORE_PATH);

        res.json({ message: `'${req.file.originalname}' processed successfully!` });
    } catch (error) {
        console.error("Error in /upload:", error);
        res.status(500).json({ error: "Failed to process file." });
    }
});

app.post('/query', async (req, res) => {
    const { query } = req.body;
    if (!query) return res.status(400).json({ error: "Query is missing." });

    try {
        await fs.access(VECTOR_STORE_PATH);

        const embeddings = new HuggingFaceInferenceEmbeddings();
        const vectorStore = await FaissStore.load(VECTOR_STORE_PATH, embeddings);
        const retriever = vectorStore.asRetriever();

        const llm = new ChatGoogleGenerativeAI({
            modelName: "gemini-pro",
            apiKey: process.env.GOOGLE_API_KEY
        });

        const prompt = ChatPromptTemplate.fromTemplate(`
            Using these documents, answer the user's question succinctly.
            Context: {context}
            Question: {input}
            Answer:
        `);

        const documentChain = await createStuffDocumentsChain({ llm, prompt });
        const retrievalChain = await createRetrievalChain({ combineDocsChain: documentChain, retriever });

        const result = await retrievalChain.invoke({ input: query });

        res.json({ answer: result.answer });
    } catch (error) {
        console.error("Error in /query:", error);
        res.status(400).json({ error: "Failed to get an answer. Please upload a document first." });
    }
});

app.listen(port, () => {
    console.log(`Backend server running at http://localhost:${port}`);
});