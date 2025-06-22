# AI Assistance Server

A FastAPI-based AI application providing chat, file processing, image generation, and web data services with advanced streaming and tool-based capabilities.

## Features

- **Real-time Chat Streaming**: AI chat with context-aware responses and streaming capabilities
- **Multi-Model Support**: Supports multiple AI clients (Transformers, Llama.cpp, OpenAI)
- **File Processing**: Upload and process various file types with vector storage
- **Vector Store**: Semantic search and context retrieval from processed files
- **Image Generation**: Generate images using Stable Diffusion
- **Web Services**: Web search and content reading via integrated tools
- **Tool-based AI**: Extensible tool system for enhanced AI interactions
- **CORS-enabled API**: Cross-origin resource sharing support
- **Static File Serving**: Serve generated outputs (images, etc.)

## Project Structure

```
src/
├── constants/         # Configuration and system prompts
│   ├── config.py      # Model configurations and paths
│   ├── file_type.py   # Supported file type definitions
│   └── system_prompts.py
├── models/           # Request/Response data models
│   ├── requests/
│   │   ├── chat_request.py
│   │   └── cancel_chat_request.py
│   └── responses/
│       ├── base_response.py
│       └── base_exception_response.py
├── routes/           # API route definitions
│   ├── chat_routes.py
│   ├── process_file_routes.py
│   └── vector_store_routes.py
├── services/         # Business logic services
│   ├── chat_service.py
│   ├── image_service.py
│   ├── process_file_service.py
│   ├── web_data_service.py
│   └── vector_store_service.py
├── utils/            # Utility functions and clients
│   ├── clients/      # AI model clients
│   │   ├── transformer_client.py
│   │   ├── llama_cpp_client.py
│   │   ├── open_ai_client.py
│   │   ├── image_pipeline_client.py
│   │   └── vector_store_client.py
│   ├── tools/        # Tool implementations
│   │   ├── tools_define.py
│   │   └── tools_helper.py
│   ├── exception.py
│   ├── stream_helper.py
│   └── timing.py
└── main.py           # Application entry point
```

## API Endpoints

### Chat API

- `POST /api/v1/chat/stream`: Stream chat responses in real-time with tool support
- `POST /api/v1/chat`: Non-streaming chat response (batched)
- `POST /api/v1/chat/cancel`: Cancel an active streaming chat session

### File Processing API

- `POST /api/v1/upload-and-process-file`: Upload and process files, store in vector DB

### Vector Store API

- `GET /api/v1/vector-store/get-all-ids`: List all vector store collection IDs
- `POST /api/v1/vector-store/inspect-collection`: Inspect a specific vector store collection

## Supported File Types

- **Documents**: PDF, DOCX
- **Images**: PNG, JPG, JPEG
- **Videos**: MP4, MOV

## AI Model Support

### LLM Clients

- **Transformers**: HuggingFace models (default: NousResearch/Hermes-3-Llama-3.1-8B)
- **Llama.cpp**: GGUF models with CUDA support
- **OpenAI**: OpenAI API integration

### Image Generation

- **Stable Diffusion**: Image generation using stable-diffusion-v1-5

### Embedding Models

- **Sentence Transformers**: all-MiniLM-L6-v2 for vector embeddings

## Tool-based AI Capabilities

- **Image Generation**: Generate images from text prompts using Stable Diffusion
- **Web Search**: Search the web using Brave Search API
- **Web Content Reading**: Fetch and process web pages using Jina API

## Error Handling

- Custom exception handling with standardized JSON responses
- Global error handlers for validation and general exceptions
- Graceful resource cleanup on application shutdown

## Static Files

- Serves generated outputs (images) from `/tmp/outputs` at `/tmp/outputs` endpoint

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

### Environment Variables

Create a `.env.local` file with the following variables:

```bash
# Web Services
jina_api_key=your_jina_api_key
brave_search_api_key=your_brave_search_api_key

# OpenAI (optional)
openai_api_key=your_openai_api_key
```

### Running the Application

#### Local Development

```bash
uvicorn main:app --reload --port 7860
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
- Server: `http://0.0.0.0:7860`

## Development

### Key Dependencies

#### Core Framework

- **FastAPI**: 0.114.0 - Web framework
- **Uvicorn**: 0.34.2 - ASGI server
- **Pydantic**: Data validation

#### AI/ML Libraries

- **Transformers**: 4.52.4 - HuggingFace models
- **Diffusers**: 0.33.1 - Stable Diffusion
- **Torch**: 2.7.0 - PyTorch
- **Accelerate**: 1.6.0 - Model optimization
- **Llama-cpp-python**: 0.3.9 - GGUF model support

#### File Processing

- **BeautifulSoup4**: 4.13.4 - HTML parsing
- **LangChain**: Community, Chroma, HuggingFace integrations
- **ChromaDB**: 0.6.3 - Vector database
- **PyMuPDF**: 1.25.1 - PDF processing

#### Utilities

- **Requests**: 2.32.3 - HTTP client
- **HuggingFace Hub**: 0.32.0 - Model management

### Model Configuration

The application supports multiple AI clients with automatic fallback:

1. **Transformers Client**: Primary for HuggingFace models
2. **Llama.cpp Client**: For GGUF models with CUDA optimization
3. **OpenAI Client**: For OpenAI API integration

### Performance Optimizations

- **CUDA Support**: GPU acceleration for AI models
- **Apple Silicon**: MPS backend support for Mac devices
- **Model Quantization**: CPU/MPS optimization
- **Caching**: Model and response caching
- **Streaming**: Real-time response streaming

### File Processing Pipeline

1. **Upload**: File validation and storage
2. **Processing**: Content extraction based on file type
3. **Embedding**: Text chunking and vector embedding
4. **Storage**: Vector database storage with metadata
5. **Retrieval**: Semantic search for context-aware responses

## License

[Add your license information here]
