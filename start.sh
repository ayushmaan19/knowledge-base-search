#!/bin/bash

# Startup script for Knowledge Base Search Engine
# Clears ChromaDB and starts both backend and frontend services

set -e

echo "🚀 Starting Knowledge Base Search Engine..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Kill any existing processes
echo "🛑 Stopping any existing services..."
pkill -f "node.*server.js" 2>/dev/null || true
pkill -f "uvicorn.*chromadb" 2>/dev/null || true
sleep 1

# Clear ChromaDB database
echo "🗑️  Clearing ChromaDB database..."
rm -rf "$PROJECT_ROOT/chroma_db"

# Clear local storage
echo "🗑️  Clearing local document storage..."
cd "$PROJECT_ROOT/backend"
echo '[]' > documents.json
echo '[]' > embeddings.json

# Start Chroma server
echo "🔄 Starting ChromaDB server..."
cd "$PROJECT_ROOT"
python3 -m uvicorn chromadb_server:app --host 127.0.0.1 --port 8000 > /tmp/chroma.log 2>&1 &
CHROMA_PID=$!
sleep 2

# Verify Chroma is running
if ! curl -s http://127.0.0.1:8000/ > /dev/null; then
  echo "❌ ChromaDB failed to start. Check /tmp/chroma.log"
  kill $CHROMA_PID 2>/dev/null || true
  exit 1
fi
echo "✅ ChromaDB running (PID: $CHROMA_PID)"

# Start Backend server
echo "🔄 Starting Backend server..."
cd "$PROJECT_ROOT/backend"
node server.js > /tmp/backend.log 2>&1 &
BACKEND_PID=$!
sleep 2

# Verify Backend is running
if ! curl -s http://127.0.0.1:5001/health > /dev/null; then
  echo "❌ Backend failed to start. Check /tmp/backend.log"
  kill $BACKEND_PID 2>/dev/null || true
  kill $CHROMA_PID 2>/dev/null || true
  exit 1
fi
echo "✅ Backend running (PID: $BACKEND_PID)"

# Start Frontend (optional - uncomment if you want)
# echo "🔄 Starting Frontend..."
# cd "$PROJECT_ROOT/frontend"
# npm run dev > /tmp/frontend.log 2>&1 &
# FRONTEND_PID=$!
# sleep 3
# echo "✅ Frontend running (PID: $FRONTEND_PID)"

echo ""
echo "✨ All services started successfully!"
echo ""
echo "📍 Services running at:"
echo "   Backend API: http://127.0.0.1:5001"
echo "   ChromaDB:    http://127.0.0.1:8000"
echo ""
echo "📝 Logs available at:"
echo "   Backend: /tmp/backend.log"
echo "   ChromaDB: /tmp/chroma.log"
echo ""
echo "💡 To stop services, run: pkill -f 'node.*server.js' && pkill -f 'uvicorn.*chromadb'"
echo "💡 To view logs: tail -f /tmp/backend.log  or  tail -f /tmp/chroma.log"
echo ""

# Keep script running (optional - remove 'wait' if you want it to exit)
wait
