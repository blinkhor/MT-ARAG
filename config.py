"""
Music Therapy RAG System Configuration
"""

import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class RAGConfig:
    """RAG System Configuration Class"""
    
    # OpenAI Configuration
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.1
    embedding_model: str = "text-embedding-3-small"
    
    # Document Processing Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval Configuration
    similarity_top_k: int = 5
    similarity_cutoff: float = 0.3
    
    # Vector Database Configuration
    chroma_db_path: str = "./chroma_db"
    collection_name: str = "music_therapy"
    
    # Data Source Configuration
    data_sources_path: str = "./docs"
    playlist_sources_path: str = "./playlist_docs"
    supported_formats: List[str] = None
    
    # Streamlit Configuration
    page_title: str = "Chat MusicTherapy"
    page_icon: str = "♩"
    layout: str = "wide"
    
    # Logging Configuration
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".json", ".jsonl"]
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables"""
        return cls(
            openai_model=os.getenv("OPENAI_MODEL", cls.openai_model),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", cls.openai_temperature)),
            embedding_model=os.getenv("EMBEDDING_MODEL", cls.embedding_model),
            chunk_size=int(os.getenv("CHUNK_SIZE", cls.chunk_size)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", cls.chunk_overlap)),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", cls.similarity_top_k)),
            similarity_cutoff=float(os.getenv("SIMILARITY_CUTOFF", cls.similarity_cutoff)),
            chroma_db_path=os.getenv("CHROMA_DB_PATH", cls.chroma_db_path),
            collection_name=os.getenv("COLLECTION_NAME", cls.collection_name),
            data_sources_path=os.getenv("DATA_SOURCES_PATH", cls.data_sources_path),
            page_title=os.getenv("PAGE_TITLE", cls.page_title),
            page_icon=os.getenv("PAGE_ICON", cls.page_icon),
            layout=os.getenv("LAYOUT", cls.layout),
            log_level=os.getenv("LOG_LEVEL", cls.log_level)
        )

@dataclass
@dataclass
class SpotifyConfig:
    """Spotify API Configuration"""
    client_id: str = "aad698be48bd4c1faf6cb71e78393ee8"
    client_secret: str = "97b52ea3d8fe4cdd9399ae3707a1b2b9"
    redirect_uri: str = "http://127.0.0.1:8501"
    scope: str = (
        "user-top-read playlist-read-private user-read-recently-played user-library-read playlist-read-collaborative"
    )

    @classmethod
    def from_env(cls) -> "SpotifyConfig":
        """支持从环境变量读取，方便部署"""
        return cls(
            client_id=os.getenv("SPOTIFY_CLIENT_ID", cls.client_id),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET", cls.client_secret),
            redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", cls.redirect_uri),
            scope=os.getenv("SPOTIFY_SCOPE", cls.scope),
        )



# Predefined prompt templates
SYSTEM_PROMPTS = {
    # 文献分析专家角色
    "literature_agent": """
You are a senior music therapy researcher with 20 years of clinical experience. 
When answering literature-related questions:

1. Evidence Hierarchy:
   - Prioritize RCTs and meta-analyses (Level I evidence)
   - Clearly distinguish between strong evidence and anecdotal reports
   - Cite PMID/DOI for all referenced studies

2. Clinical Relevance:
   - Highlight population characteristics (age, condition severity)
   - Specify intervention protocols (duration, frequency, techniques)
   - Report outcome measures with effect sizes when available

3. Critical Analysis:
   - Identify limitations of cited studies
   - Note conflicting evidence when applicable
   - Suggest clinical implications

4. Format Requirements:
   [Summary] 1-2 sentence overview
   [Evidence] Bullet points with PMID citations
   [Clinical Takeaway] Practical application guidance
   [Caveats] Important limitations/contraindications

Example Response Structure:
---
[Summary] Music therapy shows moderate efficacy (d=0.45) for reducing anxiety in preoperative patients.

[Evidence]
• PMID 12345678: 30-min individualized sessions reduced STAI scores by 32% (p<0.01)
• PMID 23456789: Group drumming showed non-inferiority to midazolam (N=150 RCT)

[Clinical Takeaway]
- Recommend 3 preoperative sessions focusing on patient-preferred music
- Combine with standard anxiolytics for high-risk cases

[Caveats]
- Limited data on long-term effects beyond 72 hours
- Contraindicated for patients with musical hallucinations
---
""",

    "playlist_agent": """
    You are a therapeutic playlist design agent.

## Your Role
- Generate **evidence-informed therapeutic playlists** for patients.
- Adapt your plan based on the **generation strategy** selected by the user:
  1. **Agent Generation based on literature recommendation (default)**  
     - Primary focus: disease alignment and therapeutic goals.  
     - Then consider genre/emotion fit.  
     - Finally refine with BPM ranges, track duration, and lyrics density (instrumental or low-lyric preferred for relaxation/sleep).  

  2. **Agent Tailored based on patient assessment**  
     - First analyze and summarize the patient’s musical preferences (Spotify history, favorite genres/artists/tracks, or query-indicated tastes).  
     - Reflect these preferences in the rationale.  
     - Select preference-aligned tracks, but cap them at **≤ 50% of the total playlist size**.  
     - Fill the remaining tracks using the literature-based logic above.  

## Rules
- Always prioritize **safety and therapeutic appropriateness** over popularity.
- If the therapeutic goal is *relaxation or sleep*, avoid tracks with dense or emotionally arousing lyrics; favor instrumental/soothing genres.
- If the therapeutic goal is *cognitive stimulation*, prefer moderate-to-fast tempo tracks (e.g., 90–120 BPM) with engaging rhythm.
- Explicitly indicate how diseases, goals, genres, and preferences influenced your selections.

## Output Structure
When you answer, follow this structure:
1. **Therapeutic goal & rationale**  
   - State the primary clinical/therapeutic objectives.  
   - Explain how literature or patient preferences guide your plan.

2. **Selection rules**  
   - Show priority order: disease/goal → genre/emotion → BPM/duration/lyrics density.  
   - In patient-preference mode, clearly mention preference contribution (≤ 50%).

3. **Session flow**  
   - Describe musical flow (e.g., warm-up → stimulation/peak → cool-down).  
   - Indicate BPM progression and emotional arc.

4. **Recommended playlist**  
   - 10–30 tracks depending on user’s chosen playlist size.  
   - Each entry: `Title — Artist (BPM / Genre / Goal / Disease)`  
   - Mark which ones are preference-aligned (if any).

5. **Safety & adaptations**  
   - Note contraindications (e.g., avoid overly stimulating tracks in anxiety, avoid sad themes in depression).  
   - Suggest how to adjust the playlist if symptoms worsen or context changes.

6. **Summary**  
   - A concise 2–3 line recap highlighting the therapeutic direction and expected benefits.

    """,
    
    "summary_prompt": """
Please summarize the following music therapy related content:

Content: {content}

Summary requirements:
1. Extract key information
2. Maintain accuracy
3. Highlight important findings
4. Be concise and clear
""",
    "expert_prompt": """
You are a senior music therapy researcher with 20 years of clinical experience. 
When answering literature-related questions:

1. Evidence Hierarchy:
   - Prioritize RCTs and meta-analyses (Level I evidence)
   - Clearly distinguish between strong evidence and anecdotal reports
   - Cite PMID/DOI for all referenced studies

2. Clinical Relevance:
   - Highlight population characteristics (age, condition severity)
   - Specify intervention protocols (duration, frequency, techniques)
   - Report outcome measures with effect sizes when available

3. Critical Analysis:
   - Identify limitations of cited studies
   - Note conflicting evidence when applicable
   - Suggest clinical implications

4. Format Requirements:
   [Summary] 1-2 sentence overview
   [Evidence] Bullet points with PMID citations
   [Clinical Takeaway] Practical application guidance
   [Caveats] Important limitations/contraindications

Example Response Structure:
---
[Summary] Music therapy shows moderate efficacy (d=0.45) for reducing anxiety in preoperative patients.

[Evidence]
• PMID 12345678: 30-min individualized sessions reduced STAI scores by 32% (p<0.01)
• PMID 23456789: Group drumming showed non-inferiority to midazolam (N=150 RCT)

[Clinical Takeaway]
- Recommend 3 preoperative sessions focusing on patient-preferred music
- Combine with standard anxiolytics for high-risk cases

[Caveats]
- Limited data on long-term effects beyond 72 hours
- Contraindicated for patients with musical hallucinations
---
""",
}

# JSON field mapping for literature records
JSON_FIELD_MAPPING = {
    # Standard field names and their possible variants
    "title": ["title", "Title", "article_title", "study_title"],
    "abstract": ["abstract", "Abstract", "summary", "description"],
    "pmid": ["pmid", "PMID", "pubmed_id", "PubMed_ID"],
    "doi": ["doi", "DOI", "digital_object_identifier"],
    "authors": ["authors", "Authors", "author", "author_list"],
    "journal": ["journal", "Journal", "publication", "source"],
    "year": ["year", "Year", "publication_year", "pub_year"],
    "keywords": ["keywords", "Keywords", "terms", "mesh_terms"],
    "methods": ["methods", "Methods", "methodology", "study_design"],
    "results": ["results", "Results", "findings", "outcomes"],
    "conclusion": ["conclusion", "Conclusion", "conclusions", "implications"],
    "population": ["population", "Population", "participants", "subjects"],
    "intervention": ["intervention", "Intervention", "treatment", "therapy_type"],
    "outcome_measures": ["outcome_measures", "measures", "assessments", "instruments"]
}

# Example questions for the interface
EXAMPLE_QUESTIONS = [
    "What are the benefits of music therapy for children with autism?",
    "How effective is music therapy in treating dementia patients?",
    "What are the main techniques used in improvisational music therapy?",
    "What qualifications and skills should music therapists have?",
    # "What is the mechanism of music therapy in psychological trauma recovery?",
    # "What are the differences between group and individual music therapy?",
    # "How effective is music therapy for patients with depression?",
    # "What are the special considerations for pediatric music therapy?",
    # "What are the prospects of music therapy in neurological rehabilitation?",
    # "How do you evaluate the effectiveness of music therapy interventions?"
]

# Error messages
ERROR_MESSAGES = {
    "no_api_key": "⚠️ Please provide a valid OpenAI API key",
    "no_data_files": "⚠️ No data files found in the specified directory",
    "file_read_error": "❌ Failed to read data files: {error}",
    "initialization_error": "❌ System initialization failed: {error}",
    "query_error": "❌ Query failed: {error}",
    "no_system": "⚠️ System not initialized, please configure and initialize the system first",
    "invalid_json": "❌ Invalid JSON format in file: {filename}",
    "empty_data": "⚠️ No valid data found in the source files"
}

# Success messages
SUCCESS_MESSAGES = {
    "system_ready": "✅ System initialization successful!",
    "files_loaded": "📄 Data files loaded successfully",
    "index_built": "🔍 Index construction completed",
    "system_initialized": "🟢 System is ready"
}

# Information messages
INFO_MESSAGES = {
    "welcome": "Hello! I'm the Music Therapy Knowledge Base assistant. You can ask any questions about music therapy, and I'll provide professional answers based on the literature.",
    "system_instruction": "👆 Please configure your API key in the sidebar to initialize the system",
    "initialization_instruction": "Please enter your API key in the sidebar and click 'Initialize System'"
}

# Default configuration instance
DEFAULT_CONFIG = RAGConfig()

DISEASE_VOCAB = {
"alzheimer": {"alzheimer", "alzheimers", "alzheimer's", "alzheimer disease", "ad"},
"dementia": {"dementia", "dementing illness", "cognitive decline"},
"stroke": {"stroke", "cerebrovascular accident", "cva", "brain attack", "ischemic stroke", "hemorrhagic stroke"},
"depression": {"depression", "major depression", "clinical depression", "mdd", "depressive disorder"},
"schizophrenia": {"schizophrenia", "schizophrenic disorder", "psychotic disorder", "schizoaffective disorder"},
"bipolar disorder": {"bipolar disorder", "bipolar", "manic depression", "manic depressive disorder"},
"conduct disorder": {"conduct disorder", "behavior disorder", "disruptive behavior disorder"},
"anxiety disorder": {"anxiety disorder", "anxiety", "generalized anxiety disorder", "gad", "panic disorder"},
"breast cancer": {"breast cancer", "breast carcinoma", "mammary cancer", "bc"},
"essential varicose veins": {"essential varicose veins", "varicose veins", "venous insufficiency", "chronic venous insufficiency"},
"obsessive-compulsive disorder": {"obsessive-compulsive disorder", "ocd", "obsessive compulsive disorder"},
"drug-resistant epilepsy": {"drug-resistant epilepsy", "refractory epilepsy", "intractable epilepsy", "treatment-resistant epilepsy"},
"lung cancer": {"lung cancer", "lung carcinoma", "bronchogenic carcinoma", "pulmonary cancer", "lc"},
"dental anxiety": {"dental anxiety", "dental phobia", "odontophobia", "dental fear"},
"mild cognitive impairment": {"mild cognitive impairment", "mci", "cognitive impairment", "age-associated memory impairment"},
"displaced nasal bone fracture": {"displaced nasal bone fracture", "nasal fracture", "broken nose", "nasal bone fracture"},
"hypertension": {"hypertension", "high blood pressure", "htn", "essential hypertension"},
"stomach cancer": {"stomach cancer", "gastric cancer", "gastric carcinoma", "stomach carcinoma"},
"esophagus cancer": {"esophagus cancer", "esophageal cancer", "oesophageal cancer", "esophageal carcinoma"},
"burn": {"burn", "burn injury", "thermal injury", "burns"},
"ankylosing spondylitis": {"ankylosing spondylitis", "as", "bamboo spine", "marie-strümpell disease"},
"parkinson's disease": {"parkinson's disease", "parkinson disease", "pd", "parkinsonism", "paralysis agitans"},
"preoperative anxiety": {"preoperative anxiety", "surgical anxiety", "preop anxiety", "anesthesia anxiety"},
"acute myocardial infarction": {"acute myocardial infarction", "ami", "heart attack", "myocardial infarction", "mi", "coronary thrombosis"}
}

GOAL_VOCAB = {
    "emotional well-being": {"emotional well-being", "emotional wellbeing", "emotional wellness", "emotional health", "emotional", "well-being", "wellbeing"},
    "cognitive skills": {"cognitive", "cognition", "cognitive skills"},
    "motor skills": {"motor", "motor skills", "motoric"},
    "physical health": {"physical health", "physical", "health", "physical wellbeing", "physical wellness"},
    "psychological/behavioral": {"psychological", "behavioral", "psychological/behavioral", "psychological behavioral", "psych behavioral"}
}