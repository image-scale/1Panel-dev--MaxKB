# Todo

## Plan
Build the project incrementally starting with core data models and utilities, then add the knowledge base features, followed by the RAG pipeline components (chunking, vectorization, search), then application/agent management, and finally the chat pipeline. Each task delivers a self-contained feature with tests.

## Tasks
- [x] Task 1: Implement user model with secure password hashing and user CRUD operations (users app)
- [x] Task 2: Implement knowledge base model with folder organization and CRUD API (knowledge bases)
- [x] Task 3: Implement document model with status tracking and document management API (documents within knowledge bases)
- [>] Task 4: Implement paragraph model for storing document segments with CRUD operations
- [ ] Task 5: Implement text chunking utilities that split text into RAG-friendly segments
- [ ] Task 6: Implement embedding storage model with vector field support for semantic storage
- [ ] Task 7: Implement vector search capabilities including embedding search, keyword search, and blend search
- [ ] Task 8: Implement problem/question model with paragraph associations for Q&A
- [ ] Task 9: Implement application/agent model with knowledge base associations and settings
- [ ] Task 10: Implement chat pipeline with context management and response generation
- [ ] Task 11: Implement model provider abstraction for LLM integration
- [ ] Task 12: Implement tag system for organizing documents and knowledge bases
