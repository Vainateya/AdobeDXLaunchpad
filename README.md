# 🎓 Adobe Resources Expert - Agentic RAG Chatbot

An intelligent chatbot that serves as an expert on Adobe certification courses and resources, built using Retrieval-Augmented Generation (RAG) with OpenAI GPT-4o mini.

## 🚀 Features

- **Expert Knowledge**: Comprehensive understanding of Adobe certification courses and resources
- **Intelligent Search**: Vector-based similarity search using ChromaDB
- **Interactive Chat**: Streamlit-based web interface for easy interaction
- **Resource Analytics**: Overview of available courses by type, level, and category
- **Context-Aware Responses**: Provides relevant information based on user queries

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Internet connection for downloading dependencies

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   - Copy `env_example.txt` to `.env`
   - Add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

4. **Get your OpenAI API key**:
   - Visit [OpenAI Platform](https://platform.openai.com/api-keys)
   - Create a new API key
   - Copy it to your `.env` file

## 🚀 Usage

1. **Run the application**:

   ```bash
   streamlit run rag_chatbot.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

3. **Start chatting** with the Adobe Resources Expert!

## 💡 Example Questions

The chatbot can answer questions like:

- "What Adobe Analytics courses are available?"
- "Tell me about Adobe Commerce certifications"
- "What are the prerequisites for Expert level certifications?"
- "How much do Adobe certifications cost?"
- "What's the difference between Professional and Expert level courses?"
- "Which courses are best for beginners?"
- "What job roles are covered by Adobe certifications?"
- "Tell me about Adobe Experience Manager courses"

## 🏗️ Architecture

### Components

1. **Vector Database (ChromaDB)**: Stores embeddings of Adobe resources for similarity search
2. **Embedding Model**: Uses SentenceTransformers for creating document embeddings
3. **LLM Integration**: OpenAI GPT-4o mini for generating responses
4. **Web Interface**: Streamlit for user interaction

### Data Flow

1. **Initialization**:

   - Load `resources.json` file
   - Create document embeddings
   - Store in ChromaDB vector database

2. **Query Processing**:

   - User submits question
   - System searches for relevant documents using vector similarity
   - Context is created from relevant documents
   - GPT-4o mini generates response based on context

3. **Response Generation**:
   - LLM processes context and user query
   - Generates comprehensive, accurate response
   - Returns formatted answer to user

## 📊 Resource Categories

The system covers various Adobe product categories:

- **Adobe Experience Cloud**: Analytics, Target, Campaign, etc.
- **Adobe Creative Cloud**: Design and creative tools
- **Adobe Document Cloud**: PDF and document management
- **Adobe Commerce**: E-commerce platform (formerly Magento)
- **Adobe Experience Manager**: Content management
- **Adobe Workfront**: Work management
- **Adobe Marketo Engage**: Marketing automation
- **Adobe Journey Optimizer**: Customer journey management
- **Adobe Real-Time CDP**: Customer data platform

## 🎯 Certification Levels

- **Foundations**: Basic introduction courses
- **Professional**: Intermediate level with hands-on experience
- **Expert**: Advanced level for experienced professionals
- **Master**: Highest level for architects and senior roles

## 🔧 Configuration

### Model Settings

- **LLM**: OpenAI GPT-4o mini
- **Embedding Model**: all-MiniLM-L6-v2
- **Vector Database**: ChromaDB with DuckDB backend
- **Max Tokens**: 1000
- **Temperature**: 0.7

### Customization

You can modify the following in `rag_chatbot.py`:

- Number of search results (`n_results` parameter)
- System prompt for different behavior
- Temperature and max_tokens for response generation
- Vector database settings

## 📁 Project Structure

```
├── rag_chatbot.py          # Main application
├── requirements.txt        # Python dependencies
├── resources.json          # Adobe resources data
├── env_example.txt         # Environment variables template
├── README.md              # This file
└── chroma_db/             # Vector database (created automatically)
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:

   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Error**: Check your API key in the `.env` file

3. **Memory Issues**: The vector database is stored locally and may grow large. You can delete the `chroma_db/` folder to reset it.

4. **Slow Performance**: First run will be slower as it creates the vector database. Subsequent runs will be faster.

### Performance Tips

- The system caches the vector database locally
- First-time setup may take a few minutes
- Subsequent queries are much faster
- You can adjust the number of search results for faster/slower responses

## 🤝 Contributing

Feel free to enhance the system by:

- Adding more sophisticated search algorithms
- Implementing conversation memory
- Adding export functionality for search results
- Creating additional visualization features
- Improving the UI/UX

## 📄 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- Adobe for providing comprehensive certification resources
- OpenAI for the GPT-4o mini model
- Streamlit for the web framework
- ChromaDB for vector storage
- SentenceTransformers for embeddings

---

**Happy Learning! 🎓**
