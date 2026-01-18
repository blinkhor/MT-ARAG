<p align="center">
  <strong>English</strong> | <a href="README_CN.md">中文说明</a>
</p>

# 🎵 Music Therapy Knowledge Base RAG System

A music therapy literature retrieval and question-answering system built with **LlamaIndex**.
It supports importing literature data from Excel files and provides intelligent, RAG-based Q&A functionality.

## ✨ Features

* 📊 **Excel Data Import**: Load music therapy literature data directly from Excel files
* 🤖 **Intelligent Q&A**: Professional music therapy question answering based on RAG technology
* 🎯 **Semantic Retrieval**: Efficient semantic search using a vector database
* 💬 **User-Friendly Interface**: Interactive web interface built with Streamlit
* 📚 **Knowledge Base Management**: Persistent storage with support for incremental updates

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd music-therapy-rag

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

#### Obtain an OpenAI API Key

1. Visit the [OpenAI website](https://openai.com/api/)
2. Register and obtain an API key
3. Ensure your account has sufficient credits

#### Prepare the Excel Data File

Make sure your Excel file contains the following columns (recommended format):

| Column Name | Description      | Example                                      |
| ----------- | ---------------- | -------------------------------------------- |
| Title       | Study title      | "Music Therapy for Autism Spectrum Disorder" |
| Author      | Authors          | "Smith, J. & Johnson, A."                    |
| Year        | Publication year | 2023                                         |
| Journal     | Journal name     | "Journal of Music Therapy"                   |
| Abstract    | Abstract         | "This study investigates..."                 |
| Keywords    | Keywords         | "autism, music therapy, intervention"        |
| DOI         | DOI              | "10.1093/jmt/thxx001"                        |
| Methods     | Research methods | "Randomized controlled trial"                |
| Results     | Results          | "Significant improvement observed"           |
| Conclusion  | Conclusion       | "Music therapy shows promise..."             |

### 3. Run the System

```bash
streamlit run main.py
```

The application will open in your browser (usually at [http://localhost:8501](http://localhost:8501)).

### 4. Usage Workflow

1. **Configure the System**

   * Enter your OpenAI API key in the sidebar
   * Upload the Excel file containing music therapy literature

2. **Initialize the Knowledge Base**

   * Click the **"Initialize System"** button
   * Wait for the vector index to be built (may take several minutes on first run)

3. **Start Chatting**

   * Enter questions in the chat interface
   * The system will generate professional answers based on the literature

## 📋 Excel File Format Requirements

### Basic Requirements

* File format: `.xlsx` or `.xls`
* The first row must contain column headers
* Each row represents one literature entry

### Recommended Column Structure

```
Title | Author | Year | Journal | Abstract | Keywords | DOI | Methods | Results | Conclusion
```

### Data Quality Recommendations

* **Completeness**: Fill in as many important fields as possible
* **Consistency**: Keep formats consistent (e.g., year format, author naming)
* **Accuracy**: Ensure the data is accurate and reliable

## 🔧 Advanced Configuration

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
CHUNK_SIZE=512
CHUNK_OVERLAP=50
SIMILARITY_TOP_K=5
SIMILARITY_CUTOFF=0.7
```

### Custom Parameters

Modify parameters in `main.py`:

```python
# Document chunking settings
node_parser = SimpleNodeParser.from_defaults(
    chunk_size=512,      # Adjust chunk size
    chunk_overlap=50     # Adjust overlap size
)

# Retrieval settings
retriever = VectorIndexRetriever(
    index=self.index,
    similarity_top_k=5   # Adjust number of retrieved chunks
)

# Similarity threshold
postprocessor = SimilarityPostprocessor(
    similarity_cutoff=0.7  # Adjust similarity cutoff
)
```

## 📚 Usage Examples

### Example Questions

* "What benefits does music therapy have for children with autism?"
* "How effective is music therapy in treating dementia?"
* "What are the main techniques used in improvisational music therapy?"
* "What qualifications are required to become a music therapist?"

### Query Tips

1. **Be specific**: More specific questions yield more accurate answers
2. **Use professional terminology**: Use domain-specific music therapy terms
3. **Ask from multiple perspectives**: Explore the same topic from different angles

## 🛠️ Troubleshooting

*(Add common issues and solutions here if needed)*

## 📦 Project Structure

```
music-therapy-rag/
├── main.py                 # Main application file
├── requirements.txt        # Dependency list
├── README.md               # Project documentation
├── .env                    # Environment variables (create manually)
├── chroma_db/              # Vector database storage
└── temp_excel.xlsx         # Temporary Excel file
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter issues or have suggestions:

1. Check the troubleshooting section
2. Search existing Issues
3. Create a new Issue with a detailed description

## 🎯 Roadmap

* [ ] Support more file formats (CSV, JSON)
* [ ] Add literature deduplication
* [ ] Support multilingual queries
* [ ] Add data visualization features
* [ ] Integrate more vector database options
* [ ] Add user management

---

**Note**: This system is intended for **academic research and educational purposes only**.
Please ensure compliance with relevant copyright and usage regulations.
