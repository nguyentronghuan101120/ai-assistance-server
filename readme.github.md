# AI Project

A FastAPI-based AI application providing chat, file, image, and web data services with advanced streaming and tool-based capabilities.

## Features

- Real-time chat streaming with AI (supports context from uploaded files)
- File upload, processing, and vector storage (PDF, DOCX, images, etc.)
- Vector store for semantic search and context retrieval
- Image generation using Stable Diffusion
- Web search and web content reading via integrated tools
- Tool-based AI interactions (extensible via `utils/tools`)
- CORS-enabled API endpoints
- Static file serving for generated outputs

## Project Structure

```
src/
├── constants/         # System prompts and configuration
│   ├── system_prompts.py
│   └── file_type.py
├── models/           # Request/Response models
│   ├── requests/
│   └── responses/
├── routes/           # API route definitions
│   ├── chat_routes.py
│   ├── process_file_routes.py
│   └── vector_store_routes.py
├── services/         # Business logic
│   ├── chat_service.py
│   ├── image_service.py
│   ├── process_file_service.py
│   ├── web_data_service.py
│   └── vector_store_service.py
├── utils/            # Utility functions
│   ├── tools/        # Tool implementations (image, web search, etc.)
│   ├── client.py
│   ├── exception.py
│   ├── image_pipeline.py
│   └── timing.py
└── main.py           # Application entry point
```

## API Endpoints

### Chat API

- `POST /api/v1/chat/stream`: Stream chat responses in real-time (supports tool-based and context-aware interactions)
- `POST /api/v1/chat`: Non-streaming chat (batched response)

### File Processing API

- `POST /api/v1/process-file/upload-and-process-file`: Upload and process files (PDF, DOCX, images, etc.), store in vector DB

### Vector Store API

- `GET /api/v1/vector-store/get-all-ids`: List all vector store collection IDs
- `POST /api/v1/vector-store/inspect-collection`: Inspect a specific vector store collection

## Tool-based AI

- **Image Generation**: Generate images from prompts using Stable Diffusion
- **Web Search**: Search the web using Brave Search API
- **Web Content Reading**: Fetch and summarize web pages using Jina API

## Error Handling

- Custom exceptions and global error handling
- Standardized JSON error responses

## Static Files

- Serves generated outputs (e.g., images) from `/outputs` at `/outputs` endpoint

## Getting Started

### Prerequisites

- Python 3.11
- CUDA 12.9.0 (for GPU acceleration)
- FastAPI 0.114.0
- Uvicorn 0.34.2

### Installation

1. Clone the repository
2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

#### Local Development

```bash
uvicorn main:app --reload --port 8080
```

#### Docker Deployment

```bash
# Build the Docker image
docker build -t ai-assistance-server .

# Run the container
docker run -p 7860:7860 --gpus all ai-assistance-server
```

The application will be available at:

- Local: `http://localhost:7860`
- Server: `http://0.0.0.0:7860` or https://leonguyen101120-ai-assistance.hf.space

## Development

### Key Dependencies

- **AI/ML**:

  - diffusers 0.33.1
  - transformers 4.52.4
  - torch 2.7.0
  - accelerate 1.6.0

- **File Processing** (Optional):
  - beautifulsoup4 4.13.4
  - langchain_chroma 0.2.2
  - langchain_huggingface 0.1.2
  - langchain_community 0.3.19
  - chromadb 0.6.3
  - pymupdf 1.25.1

### Environment Variables

The following environment variables are required for specific features:

- Brave Search API key (for web search)
- Jina API key (for web content reading)
- HuggingFace API key (for model access)

## License

[Add your license information here]
