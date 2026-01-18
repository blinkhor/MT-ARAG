import os
import json
import streamlit as st
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("llama_index").setLevel(logging.DEBUG)
for noisy in ("urllib3", "chromadb", "chromadb.telemetry"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

import hashlib
from typing import List, Dict, Optional, Any
from datetime import datetime
import time
from pathlib import Path
import glob
import re

import httpx

# --- LlamaIndex Core ---
from llama_index.core import (
    VectorStoreIndex,
    Document,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.llms import ChatMessage

# --- Providers: OpenAI & Chroma ---
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
debug_handler = LlamaDebugHandler(print_trace_on_end=True)
Settings.callback_manager = CallbackManager([debug_handler])

# spotify
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Import configuration
from config import (
    RAGConfig, SpotifyConfig, SYSTEM_PROMPTS, JSON_FIELD_MAPPING,
    EXAMPLE_QUESTIONS, ERROR_MESSAGES, SUCCESS_MESSAGES, INFO_MESSAGES,
    DISEASE_VOCAB, GOAL_VOCAB
)


# SYSTEM_PROMPTS.setdefault("literature_agent", """You are a meticulous music-therapy literature analyst.
# - Cite key findings, designs, populations, outcomes.
# - Summarize consensus, uncertainties, contraindications.
# - Explain how evidence informs practice succinctly.
# """)
# SYSTEM_PROMPTS.setdefault("playlist_agent", """You design evidence-informed therapeutic playlists.
# - Personalize by patient preference (artists/genres), emotion goals, and BPM ranges.
# - If provided, use Spotify top artists/tracks as seeds; otherwise propose plausible seeds.
# - Output a structured plan: goal → selection criteria (genre/emotion/BPM) → track suggestions.
# - Prefer instrumental/low-lyric density for relaxation use-cases; escalate/descend tempo when appropriate.
# """)
# SYSTEM_PROMPTS.setdefault("expert_prompt", SYSTEM_PROMPTS.get("expert_prompt", ""))

# httpx 新版使用 proxy=（不是 proxies=）
http_client = httpx.Client(
    timeout=60,
    trust_env=False,
)

# ========== 工具函数：稳定 node_id，避免重复嵌入 ==========
def make_node_id(doc_id: str, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{doc_id}::{h}"

def init_session_state():
    defaults = {
        "spotify_authed": False,
        "spotify_prefs": None,
        "spotify_error": None,
        "spotify_code_consumed": False,  # 回跳后是否已消费 code
        "messages": [{"role": "assistant", "content": INFO_MESSAGES["welcome"]}],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def get_query_param(name: str) -> Optional[str]:
    try:
        return st.query_params.get(name)  # 新版 Streamlit
    except Exception:
        params = st.experimental_get_query_params()
        v = params.get(name)
        return v[0] if isinstance(v, list) and v else None

def clear_query_params():
    try:
        st.query_params.clear()
    except Exception:
        st.experimental_set_query_params()
        
def _norm_tokens_lower(s: str) -> List[str]:
    return re.findall(r'[a-zA-Z0-9\u4e00-\u9fa5]+', (s or "").lower())


# =========================
#         核心类
# =========================
class MusicTherapyRAG:
    def __init__(self, openai_api_key: str, config: RAGConfig = None):
        """
        Initialize Music Therapy RAG System (双域：文献索引 + 歌单内存)
        """
        self.config = config or RAGConfig()
        self.openai_api_key = openai_api_key

        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        self.logger = logging.getLogger(__name__)

        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # --- LLM & Embedding：使用全局 Settings ---
        self.llm = OpenAI(
            model=self.config.openai_model,
            api_base="https://api.vveai.com/v1",
            temperature=self.config.openai_temperature,
            http_client=http_client,
            timeout=120,
            max_retries=5,
        )
        self.embed_model = OpenAIEmbedding(
            model=self.config.embedding_model,
            api_base="https://api.vveai.com/v1",
            http_client=http_client,
            embed_batch_size=16,
        )
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # --- 向量数据库（Chroma 持久化） — 文献域 ---
        self.chroma_client = chromadb.PersistentClient(path=self.config.chroma_db_path)
        self.chroma_collection_lit = self.chroma_client.get_or_create_collection(
            f"{self.config.collection_name}_literature"
        )
        self.vector_store_lit = ChromaVectorStore(chroma_collection=self.chroma_collection_lit)

        # --- 存储上下文 ---
        self.storage_lit = StorageContext.from_defaults(vector_store=self.vector_store_lit)

        # 双域索引与引擎
        self.index_lit = None
        self.query_engine_lit = None

        # 歌单域：不建索引，使用内存目录
        self.playlist_catalog: List[Dict[str, Any]] = []
        self.index_pl = None
        self.query_engine_pl = None

        self.document_count = 0  # 仅统计文献域
        
    def _chat(self, llm, system_text: str, user_text: str, temperature: float = 0.0) -> str:
        messages = [
            ChatMessage(role="system", content=system_text),
            ChatMessage(role="user", content=user_text),
        ]
        resp = llm.chat(messages, temperature=temperature)
        return getattr(resp, "message", resp).content if hasattr(resp, "message") else str(resp)

    # ========== 字段规范化 ==========
    def normalize_field_name(self, field_name: str) -> Optional[str]:
        field_name = field_name.strip()
        for standard_name, variants in JSON_FIELD_MAPPING.items():
            if field_name in variants:
                return standard_name
        return None

    # ========== 文件加载 ==========
    def load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        data = []
        file_path = Path(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                record = json.loads(line)
                                data.append(record)
                            except json.JSONDecodeError as e:
                                self.logger.warning(f"Invalid JSON in {file_path} line {line_num}: {e}")
                                continue
                elif file_path.suffix == '.json':
                    content = json.load(f)
                    if isinstance(content, list):
                        data.extend(content)
                    elif isinstance(content, dict):
                        data.append(content)
                    else:
                        self.logger.warning(f"Unexpected JSON structure in {file_path}")
            self.logger.info(f"Loaded {len(data)} records from {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            raise

    def load_all_data_files(self) -> List[Dict[str, Any]]:
        all_data = []
        data_dir = Path(self.config.data_sources_path)
        if not data_dir.exists():
            self.logger.warning(f"Data sources directory {data_dir} does not exist")
            return all_data
        json_files = list(data_dir.glob("*.json"))
        jsonl_files = list(data_dir.glob("*.jsonl"))
        all_files = json_files + jsonl_files
        if not all_files:
            self.logger.warning(f"No JSON/JSONL files found in {data_dir}")
            return all_data
        self.logger.info(f"Found {len(all_files)} data files")
        for file_path in all_files:
            try:
                file_data = self.load_json_file(file_path)
                all_data.extend(file_data)
            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")
                continue
        self.logger.info(f"Total records loaded: {len(all_data)}")
        return all_data

    def create_document_from_record(self, record: Dict[str, Any], doc_id: str) -> Optional[Document]:
        try:
            normalized_record = {}
            for field, value in record.items():
                normalized_field = self.normalize_field_name(field)
                if normalized_field:
                    normalized_record[normalized_field] = value
                else:
                    normalized_record[field] = value

            title = normalized_record.get('title', '')
            abstract = normalized_record.get('abstract', '')

            if not title and not abstract:
                self.logger.warning(f"Record {doc_id} has no title or abstract, skipping")
                return None

            text_parts = []
            if title:
                text_parts.append(f"Title: {title}")
            if abstract:
                text_parts.append(f"Abstract: {abstract}")
            text_content = "\n\n".join(text_parts)

            metadata = {"doc_id": doc_id}

            metadata_fields = ['PMID', 'Title', 'Authors', 'Journal/Book', 'Publication Year',
                               'StudyDesign', 'SampleSize', 'Disease_primary category', 'Provider/Platform',
                               'InterventionType', 'Theme', 'Music selection strategy', 'Duration', 'Frequency', 'Study Period',
                               'Intervention Node', 'Intensity']
            for field in metadata_fields:
                if field in normalized_record:
                    value = normalized_record[field]
                    if value is not None:
                        if isinstance(value, list):
                            metadata[field] = "; ".join(str(v) for v in value if v)
                        else:
                            metadata[field] = str(value).strip()

            doc = Document(
                text=text_content,
                metadata=metadata,
                doc_id=doc_id
            )
            return doc
        except Exception as e:
            self.logger.warning(f"Error creating document from record {doc_id}: {e}")
            return None

    def load_literature_data(self) -> List[Document]:
        try:
            all_records = self.load_all_data_files()
            if not all_records:
                raise ValueError("No data records found")
            documents = []
            for index, record in enumerate(all_records):
                doc_id = f"doc_{index}"
                if 'pmid' in record or 'PMID' in record:
                    pmid = record.get('pmid') or record.get('PMID')
                    if pmid:
                        doc_id = f"pmid_{pmid}"
                doc = self.create_document_from_record(record, doc_id)
                if doc:
                    documents.append(doc)
            if not documents:
                raise ValueError("No valid documents created from the data")
            self.document_count = len(documents)
            self.logger.info(f"Successfully created {len(documents)} documents")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading literature data: {e}")
            raise

    # ========== 歌单：零 embedding，加载到内存 ==========
    def load_playlist_docs(self) -> List[Document]:
        """
        读取 song_*.json / jsonl，填充 self.playlist_catalog（结构化），
        并返回最小 Doc（供可视化/统计，不参与向量检索）
        """
        base = Path(self.config.playlist_sources_path)
        docs = []
        catalog = []
        if not base.exists():
            self.logger.warning(f"Playlist data dir not found: {base}")
            return docs
        files = list(base.glob("song_[0-9].json")) + list(base.glob("song_[0-9][0-9].json")) + list(base.glob("song_[0-9][0-9][0-9].json"))
        for fp in files:
            try:
                for record in self.load_json_file(fp):
                    # --- Genre 规范化 ---
                    raw_genre = record.get("Genre")
                    genres = []
                    if isinstance(raw_genre, list):
                        for g in raw_genre:
                            if isinstance(g, str):
                                # 可能是 "Pop, Oldies, Traditional Pop"
                                parts = [x.strip() for x in g.split(",") if x.strip()]
                                genres.extend(parts)
                    elif isinstance(raw_genre, str):
                        genres = [x.strip() for x in raw_genre.split(",") if x.strip()]
                    genres = list(dict.fromkeys(genres))  # 去重并保序

                    # --- Goal / Disease 规范化 ---
                    def _split_clean_list(val, seps=(";", "；", ",")):
                        out = []
                        if isinstance(val, list):
                            for it in val:
                                if isinstance(it, str):
                                    s = it
                                else:
                                    s = str(it or "")
                                for sp in seps:
                                    s = s.replace(sp, "|")
                                out.extend([x.strip() for x in s.split("|") if x.strip()])
                        elif isinstance(val, str):
                            s = val
                            for sp in seps:
                                s = s.replace(sp, "|")
                            out.extend([x.strip() for x in s.split("|") if x.strip()])
                        return list(dict.fromkeys(out))

                    goals   = _split_clean_list(record.get("Goal-Orientation"))
                    diseases = _split_clean_list(record.get("Disease information"))

                    lyrics_flag = record.get("Lyrics")
                    if isinstance(lyrics_flag, str):
                        lyrics_flag = lyrics_flag.strip().lower() in ("yes", "true", "1")

                    entry = {
                        "doc_id": f"pl_{fp.stem}_{hashlib.md5(str(record.get('No.', '')).encode()).hexdigest()[:8]}",
                        "title": record.get("Song Title", "") or "",
                        "album": record.get("Album", "") or "",
                        "artist": record.get("Artist", "") or "",
                        "file_name": record.get("File Name", "") or "",
                        "lyrics": record.get("Lyrics", "") or "",
                        "lyrics_flag": bool(lyrics_flag),  # NEW: 是否有歌词
                        "genre": genres,                    # CHANGED: 规范化后的列表
                        "duration": float(record.get("Duration", 0) or 0),
                        "bpm": float(record.get("BPM", 0) or 0),
                        "goals": goals,                     # NEW
                        "diseases": diseases,               # NEW
                        "track_number": record.get("No.", 0) or 0,
                        "source_file": str(fp.name),
                    }
                    catalog.append(entry)
                    docs.append(Document(text="Music metadata document", metadata=entry))
            except Exception as e:
                self.logger.error(f"Failed to load playlist doc {fp}: {e}")
        self.playlist_catalog = catalog
        self.logger.info(f"Loaded {len(docs)} playlist docs; catalog size={len(self.playlist_catalog)}")
        return docs

    # ========== 通用增量索引构建（文献域用） ==========
    def build_or_update_index(self, documents: List[Document], storage_ctx: StorageContext,
                              vector_store: ChromaVectorStore) -> VectorStoreIndex:
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        for n in nodes:
            doc_id = n.metadata.get("doc_id", "unknown")
            n.id_ = make_node_id(doc_id, n.get_content())
            n.metadata["doc_id"] = doc_id

        coll = vector_store._collection
        existing_count = coll.count()
        if existing_count == 0:
            idx = VectorStoreIndex(nodes, storage_context=storage_ctx, show_progress=True)
        else:
            idx = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_ctx)
            existing_ids = set(coll.get().get("ids", []))
            missing = [n for n in nodes if n.id_ not in existing_ids]
            if missing:
                idx.insert_nodes(missing, show_progress=True)
        return idx

    # ========== 构建混合检索器（若无 BM25 则回落纯向量） ==========
    def _build_hybrid_retriever(self, index: VectorStoreIndex, nodes_source_docs: List[Document] = None,
                                top_k: int = None):
        top_k = top_k or self.config.similarity_top_k
        try:
            from llama_index.core.retrievers import BM25Retriever, EnsembleRetriever
            bm25 = None
            if nodes_source_docs:
                bm25 = BM25Retriever.from_defaults(documents=nodes_source_docs, similarity_top_k=top_k)
            vec = index.as_retriever(similarity_top_k=top_k)
            if bm25 is None:
                return vec
            return EnsembleRetriever(retrievers=[vec, bm25], weights=[0.6, 0.4])
        except Exception:
            return index.as_retriever(similarity_top_k=top_k)

    # ========== 设置查询引擎 ==========
    def setup_query_engines(self):
        post = SimilarityPostprocessor(similarity_cutoff=self.config.similarity_cutoff)
        if self.index_lit:
            r_lit = self._build_hybrid_retriever(self.index_lit, None, self.config.similarity_top_k)
            self.query_engine_lit = RetrieverQueryEngine(retriever=r_lit, node_postprocessors=[post])

    # ========== 初始化 ==========
    def initialize_system(self):
        """
        Initialize the entire RAG system
        """
        try:
            self.logger.info("Starting Music Therapy RAG system initialization...")
            # 文献域：索引（向量/混合）
            lit_docs = self.load_literature_data()
            self.index_lit = self.build_or_update_index(lit_docs, self.storage_lit, self.vector_store_lit)

            # 歌单域：零 embedding，仅加载内存
            _ = self.load_playlist_docs()

            self.setup_query_engines()
            self.logger.info("Music Therapy RAG system initialization completed!")
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            raise

    # ========== 意图识别 ==========
    def classify_intent(self, user_query: str) -> str:
        system = (
            "You are an intent classifier for a music-therapy app. "
            "Classify the user's query into exactly one of: "
            "'literature' (asks about evidence, papers, RCTs, clinical findings, mechanisms), "
            "'playlist' (asks to recommend songs/playlists, match patient preferences, BPM/genre/emotion, Spotify interactions). "
            "Return JSON with fields: intent, reason."
        )
        prompt = f"User query: {user_query}\nReturn JSON only."

        # 尝试 LLM JSON 输出；失败则关键词兜底
        try:
            txt = self._chat(self.llm, system, prompt, temperature=0.5).strip()
            data = json.loads(txt) if txt.startswith("{") else json.loads(txt[txt.find("{"):txt.rfind("}")+1])
            intent = (data or {}).get("intent")
            if intent in ("literature", "playlist"):
                return intent
        except Exception:
            pass

        q = user_query.lower()
        if any(k in q for k in ["歌单", "playlist", "推荐", "bpm", "spotify", "曲目", "歌曲", "播放列表"]):
            return "playlist"
        return "literature"

    # ========== 歌单检索 ==========
    
    def _extract_disease_goal_from_query(self, q: str) -> Dict[str, List[str]]:  # NEW
        toks = _norm_tokens_lower(q)
        diseases, goals = set(), set()
        for t in toks:
            for k, vs in DISEASE_VOCAB.items():
                if t in vs:
                    diseases.add(k)
            for k, vs in GOAL_VOCAB.items():
                if t in vs:
                    goals.add(k)
        return {"q_diseases": list(diseases), "q_goals": list(goals)}
    
    def _build_preference_profile(self, q: str) -> Dict[str, Any]:  # NEW
        """
        先用 Spotify 偏好（若已授权），否则回退到用户 query 中提取的流派/艺人关键词。
        """
        prefs = st.session_state.get("spotify_prefs") or {}
        profile = {"genres": set(), "artists": set(), "tracks": set(), "has_spotify": bool(prefs)}

        if prefs:  # Spotify
            for a in (prefs.get("top_artists_short") or []):
                n = (a.get("name") or "").strip()
                if n: profile["artists"].add(n.lower())
            for t in (prefs.get("top_tracks_short") or []):
                n = (t.get("name") or "").strip()
                if n: profile["tracks"].add(n.lower())
            # 也可统计 top_tracks 的主流派（依赖额外调用，这里略）

        # 从用户话术补充：流派/艺人词
        toks = _norm_tokens_lower(q)
        # 简单将中文映射至英文常见 genre
        for t in toks:
            if t in GENRE_CN2EN:
                profile["genres"].add(GENRE_CN2EN[t])
            else:
                # 直接收进潜在 genre（与曲目 genre 交集时生效）
                if t in {"classical","instrumental","ambient","new","age","world","pop","jazz","lofi","meditation","nature"}:
                    profile["genres"].add(t if t!="new" else "new age")
        # 艺人：粗略策略——若出现大写/标题式词可以进一步白名单，这里保持简单
        # 你也可以在此接一个“已知艺人清单”匹配

        return {
            "genres": list(profile["genres"]),
            "artists": list(profile["artists"]),
            "tracks": list(profile["tracks"]),
            "has_spotify": profile["has_spotify"]
        }
    
    def _parse_playlist_constraints(self, q: str) -> Dict[str, Any]:
        ql = q.lower()

        bpm_min, bpm_max = None, None
        m = re.search(r'(\d{2,3})\s*-\s*(\d{2,3})\s*bpm', ql)
        if m:
            bpm_min, bpm_max = int(m.group(1)), int(m.group(2))
        else:
            m2 = re.search(r'(\d{2,3})\s*bpm', ql)
            if m2:
                bpm_min = int(m2.group(1)) - 5
                bpm_max = int(m2.group(1)) + 5

        # 单位更宽松：min|mins|minute|minutes|分钟|分
        dur_max = None
        m = re.search(r'(?:<=|小于|少于|不超过)\s*(\d{1,2})\s*(?:min|mins|minute|minutes|分钟|分)\b', ql)
        if m:
            dur_max = int(m.group(1)) * 60

        genre_vocab = ["classical","instrumental","ambient","new age","world","pop","jazz","lofi","meditation","nature"]
        emotion_vocab = ["relax","relaxing","calm","soothing","focus","sleep","anxious","sad","melancholy","happy","energetic"]
        want_genres = [g for g in genre_vocab if g in ql]
        want_emotions = [e for e in emotion_vocab if e in ql]

        tokens = re.findall(r'[a-zA-Z0-9\u4e00-\u9fa5]+', q)
        keywords = [t for t in tokens if len(t) >= 2]

        dg = self._extract_disease_goal_from_query(q)  # NEW
        return {
            "bpm_min": bpm_min, "bpm_max": bpm_max,
            "dur_max": dur_max,
            "want_genres": want_genres,
            "want_emotions": want_emotions,
            "keywords": keywords,
            "q_diseases": dg["q_diseases"],  # NEW
            "q_goals": dg["q_goals"],        # NEW
        }

    # ====== Literature 优先打分 ======
    def _score_literature(self, track: Dict[str, Any], cons: Dict[str, Any]) -> float:
        title = (track.get("title") or "").lower()
        artist = (track.get("artist") or "").lower()
        album = (track.get("album") or "").lower()
        lyrics_text = (track.get("lyrics") or "").lower()
        genres = [g.lower() for g in (track.get("genre") or [])]
        bpm = float(track.get("bpm") or 0)
        dur = float(track.get("duration") or 0)
        has_lyrics = bool(track.get("lyrics_flag", False))

        # 硬过滤：BPM & 时长
        if cons["bpm_min"] is not None and bpm < cons["bpm_min"]:
            return -1.0
        if cons["bpm_max"] is not None and bpm > cons["bpm_max"]:
            return -1.0
        if cons["dur_max"] is not None and dur and dur > cons["dur_max"]:
            return -1.0

        score = 0.0

        # 1) 疾病/目标（最高权重）
        if cons["q_diseases"]:
            overlap = len(set([x.lower() for x in (track.get("diseases") or [])]) & set(cons["q_diseases"]))
            score += 0.35 * min(1.0, overlap / max(1, len(cons["q_diseases"])))
        if cons["q_goals"]:
            overlap = len(set([x.lower() for x in (track.get("goals") or [])]) & set(cons["q_goals"]))
            score += 0.25 * min(1.0, overlap / max(1, len(cons["q_goals"])))

        # 2) genre
        if cons["want_genres"]:
            overlap = len(set(cons["want_genres"]) & set(genres))
            score += 0.15 * min(1.0, overlap / max(1, len(cons["want_genres"])))

        # 3) emotion / keyword 命中
        hits = 0
        for kw in cons["keywords"] + cons["want_emotions"]:
            kwl = kw.lower()
            if kwl in title or kwl in artist or kwl in album or (lyrics_text and kwl in lyrics_text):
                hits += 1
        if hits:
            score += 0.1 * min(1.0, hits / 5.0)

        # 4) Relax/Sleep → 低歌词偏好
        if cons["q_goals"] and any(g in {"relaxation", "sleep"} for g in cons["q_goals"]):
            if not has_lyrics:
                score += 0.1
            else:
                score -= 0.05

        return max(0.0, min(1.0, score))

    # ====== 偏好优先打分（艺人/曲目/流派） ======
    def _score_preference(self, track: Dict[str, Any], pref: Dict[str, Any]) -> float:
        title = (track.get("title") or "").lower()
        artist = (track.get("artist") or "").lower()
        genres = [g.lower() for g in (track.get("genre") or [])]

        s = 0.0
        if pref.get("artists") and artist in pref["artists"]:
            s += 0.5
        if pref.get("tracks") and title in pref["tracks"]:
            s += 0.3
        if pref.get("genres") and (set(pref["genres"]) & set(genres)):
            s += 0.2
        return max(0.0, min(1.0, s))

    def _score_track(self, track: Dict[str, Any], constraints: Dict[str, Any], prefs: Optional[dict]) -> float:
        title = (track.get("title") or "").lower()
        artist = (track.get("artist") or "").lower()
        album = (track.get("album") or "").lower()
        lyrics = (track.get("lyrics") or "").lower()
        genres = [g.lower() for g in (track.get("genre") or [])]

        # 硬过滤
        bpm = track.get("bpm") or 0
        if constraints["bpm_min"] is not None and bpm < constraints["bpm_min"]:
            return -1.0
        if constraints["bpm_max"] is not None and bpm > constraints["bpm_max"]:
            return -1.0
        if constraints["dur_max"] is not None:
            dur = track.get("duration") or 0
            if dur and dur > constraints["dur_max"]:
                return -1.0

        score = 0.0

        # 类别匹配（genre）
        if constraints["want_genres"]:
            overlap = len(set(constraints["want_genres"]) & set(genres))
            score += 0.15 * min(1.0, overlap / max(1, len(constraints["want_genres"])))

        # 关键词命中（标题/歌手/专辑/歌词）
        hits = 0
        for kw in constraints["keywords"] + constraints["want_emotions"]:
            kwl = kw.lower()
            if kwl in title or kwl in artist or kwl in album or (lyrics and kwl in lyrics):
                hits += 1
        if hits:
            score += 0.3 * min(1.0, hits / 5.0)

        # 偏好加权（UI + Spotify）
        bonus = 0.0
        ui_genres = [g.lower() for g in (st.session_state.get("genre_strategy") or [])]
        ui_emotions = [e.lower() for e in (st.session_state.get("emotion_strategy") or [])]
        if ui_genres and (set(ui_genres) & set(genres)):
            bonus += 0.1
        if ui_emotions:
            emo_hits = sum(1 for e in ui_emotions if e in title or e in album or (lyrics and e in lyrics))
            if emo_hits:
                bonus += 0.05

        prefs = prefs or st.session_state.get("spotify_prefs")
        if prefs:
            top_artists = [a.get("name","").lower() for a in (prefs.get("top_artists_short") or [])]
            if artist in top_artists:
                bonus += 0.15
            top_tracks_names = [t.get("name","").lower() for t in (prefs.get("top_tracks_short") or [])]
            if title in top_tracks_names:
                bonus += 0.05

        score += bonus
        return max(0.0, min(1.0, score))

    def search_playlist_catalog(self, user_query: str, top_k: int = 20, strategy: str = "literature") -> List[Dict[str, Any]]:
        if not self.playlist_catalog:
            return []

        cons = self._parse_playlist_constraints(user_query)
        strategy = (strategy or "").lower()

        if "patient" in strategy:  # Agent Tailored based on patient assessment
            pref = self._build_preference_profile(user_query)

            # 1) 偏好优先池
            pref_scored = []
            for tr in self.playlist_catalog:
                ps = self._score_preference(tr, pref)
                if ps > 0:
                    # 进一步根据“文献规则”做基本适配（过滤 & 微调）
                    ls = self._score_literature(tr, cons)
                    if ls >= 0:
                        pref_scored.append(((ps * 0.7 + ls * 0.3), tr))
            pref_scored.sort(key=lambda x: x[0], reverse=True)

            take_pref = min(len(pref_scored), max(1, top_k // 2))  # ≤ 一半
            picked = [t for _, t in pref_scored[:take_pref]]

            # 2) 文献补齐池（去除已选）
            rest_scored = []
            picked_ids = {id(p) for p in picked}
            for tr in self.playlist_catalog:
                if id(tr) in picked_ids:
                    continue
                s = self._score_literature(tr, cons)
                if s >= 0:
                    rest_scored.append((s, tr))
            rest_scored.sort(key=lambda x: x[0], reverse=True)

            fill_needed = top_k - len(picked)
            picked.extend([t for _, t in rest_scored[:max(0, fill_needed)]])
            return picked

        else:  # literature 默认
            scored = []
            for tr in self.playlist_catalog:
                s = self._score_literature(tr, cons)
                if s >= 0:
                    scored.append((s, tr))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [t for _, t in scored[:top_k]]

    # ========== 歌单 Agent 生成 ==========
    def generate_playlist_plan(self, user_query: str, candidates: List[Dict[str, Any]], strategy: str = "literature", playlist_size: int = 20) -> str:
        sys = SYSTEM_PROMPTS["playlist_agent"]
        cand_view = [
            {
                "title": c.get("title",""),
                "artist": c.get("artist",""),
                "album": c.get("album",""),
                "bpm": c.get("bpm", 0),
                "genre": c.get("genre", []),
                "duration_sec": c.get("duration", 0),
                "goals": c.get("goals", []),
                "diseases": c.get("diseases", []),
            } for c in candidates[:max(10, min(30, playlist_size*2))]
        ]
        ui_ctx = {
            "spotify_authed": st.session_state.get("spotify_authed"),
            "playlist_strategy": strategy,
            "playlist_size": playlist_size,
        }
        pref_profile = self._build_preference_profile(user_query) if "patient" in (strategy or "").lower() else {}
        cons = self._parse_playlist_constraints(user_query)

        prompt = (
            f"User query:\n{user_query}\n\n"
            f"Detected diseases/goals from query:\n{json.dumps({'diseases': cons['q_diseases'], 'goals': cons['q_goals']}, ensure_ascii=False)}\n\n"
            f"Strategy: {strategy} | Target playlist size: {playlist_size}\n"
            f"User/UI/Context:\n{json.dumps(ui_ctx, ensure_ascii=False)}\n\n"
            f"Preference profile (if any):\n{json.dumps(pref_profile, ensure_ascii=False)}\n\n"
            f"Candidate tracks (trimmed):\n{json.dumps(cand_view, ensure_ascii=False)}\n\n"
            "Task: Using the strategy, candidates and context, propose an evidence-informed therapeutic playlist.\n"
            "Rules:\n"
            "- If strategy is 'Agent Generation based on literature recommendation': prioritize disease/goal alignment, then genre/emotion, then BPM/lyrics density.\n"
            "- If strategy is 'Agent Tailored based on patient assessment': first reflect user's preferences (genres/artists/tracks) in the rationale; include preference-aligned tracks but cap them at at most HALF of the playlist size; fill the rest by literature-driven matching.\n"
            "- Prefer instrumental/low-lyric density for relaxation/sleep goals.\n"
            "Structure your answer as:\n"
            "1) Therapeutic goal & rationale\n"
            "2) Selection rules (priority order and why)\n"
            "3) Session flow (e.g., warm-up → peak → cool-down) with BPM progression\n"
            f"4) Recommended tracks ({min(10, playlist_size)}-{playlist_size}), each with: Title — Artist (BPM / Genre / Goal / Disease)\n"
            "5) Safety/contraindications & how to adapt\n"
            "6) If patient-preference strategy: explicitly state what proportion are preference-aligned (≤ 50%)\n"
        )
        try:
            txt = self._chat(self.llm, sys, prompt, temperature=0.5).strip()
            return str(txt)
        except Exception as e:
            self.logger.error(f"Playlist plan failed: {e}")
            return "Failed to generate playlist plan."


    # ========== 总查询：路由 ==========
    def query(self, question: str) -> str:
        if not (self.query_engine_lit or self.playlist_catalog):
            raise ValueError("Engines not initialized.")

        intent = self.classify_intent(question)

        if intent == "literature":
            if not self.query_engine_lit:
                return "Literature engine not ready."
            enhanced = SYSTEM_PROMPTS["literature_agent"] + "\n" + SYSTEM_PROMPTS["expert_prompt"] + f"\n\nUser Question: {question}"
            try:
                t0 = time.time()
                ans = self.query_engine_lit.query(enhanced)
                self.logger.info(f"[literature] {time.time()-t0:.2f}s")
                return str(ans)
            except Exception as e:
                self.logger.error(f"Literature query failed: {e}")
                return f"Error: {e}"

        # playlist
        strategy = (st.session_state.get("playlist_strategy") or "Agent Generation based on literature recommendation")
        playlist_size = int(st.session_state.get("playlist_size") or 20)

        candidates = self.search_playlist_catalog(question, top_k=playlist_size, strategy=strategy)
        plan = self.generate_playlist_plan(question, candidates, strategy=strategy, playlist_size=playlist_size)

        if candidates:
            lines = ["\n\n**Candidate pool (top ranked):**"]
            for i, c in enumerate(candidates[:playlist_size], 1):
                g = ", ".join(c.get("genre") or [])
                dg = "; ".join(c.get("diseases") or [])
                gl = "; ".join(c.get("goals") or [])
                lines.append(f"{i}. {c.get('title','Unknown')} — {c.get('artist','')} (BPM {int(c.get('bpm') or 0)}; {g}; Goal: {gl}; Disease: {dg})")
            return plan + "\n" + "\n".join(lines)
        return plan


    # ========== 统计 ==========
    def get_system_stats(self) -> Dict:
        data_dir = Path(self.config.data_sources_path)
        file_count = 0
        if data_dir.exists():
            file_count = len(list(data_dir.glob("*.json")) + list(data_dir.glob("*.jsonl")))
        chroma_vectors = 0
        try:
            chroma_vectors = self.chroma_collection_lit.count()
        except Exception:
            pass
        return {
            "document_count": self.document_count,
            "data_files_count": file_count,
            "index_exists": self.index_lit is not None,
            "query_engine_ready": self.query_engine_lit is not None,
            "chroma_vectors": chroma_vectors,
            "playlist_tracks": len(self.playlist_catalog),
            "config": {
                "chunk_size": self.config.chunk_size,
                "similarity_top_k": self.config.similarity_top_k,
                "similarity_cutoff": self.config.similarity_cutoff,
                "data_sources_path": self.config.data_sources_path,
                "playlist_sources_path": getattr(self.config, "playlist_sources_path", "playlist_docs"),
            }
        }

# =========================
#       Spotify 辅助
# =========================
def render_tracks(tracks, title, top_n=None):
    if not tracks:
        return
    st.markdown(f"**🎵 {title}**")
    if top_n:
        tracks = tracks[:top_n]
    for i, track in enumerate(tracks, 1):
        if isinstance(track, dict) and "track" in track and isinstance(track["track"], dict):
            track = track["track"]
        if not track or "name" not in track:
            continue
        artists = ", ".join([a.get("name", "") for a in track.get("artists", []) if a])
        st.write(f"{i}. {track.get('name', 'Unknown')} — {artists}")

def render_artists(artists, title, top_n=None):
    if not artists:
        return
    st.markdown(f"**🎤 {title}**")
    if top_n:
        artists = artists[:top_n]
    for i, artist in enumerate(artists, 1):
        st.write(f"{i}. {artist.get('name','Unknown')}（流行度: {artist.get('popularity','-')}）")

def fetch_spotify_preferences():
    sp_config = SpotifyConfig()
    sp_oauth = SpotifyOAuth(
        client_id=sp_config.client_id,
        client_secret=sp_config.client_secret,
        redirect_uri=sp_config.redirect_uri,
        scope=sp_config.scope,
        cache_path=".cache-streamlit",
        open_browser=False,
        show_dialog=True
    )

    code = get_query_param("code")

    if not code:
        auth_url = sp_oauth.get_authorize_url()
        st.markdown(
            f"<meta http-equiv='refresh' content='0; url={auth_url}'>",
            unsafe_allow_html=True
        )
        st.stop()

    token_info = sp_oauth.get_access_token(code)
    access_token = token_info["access_token"] if isinstance(token_info, dict) else token_info

    sp = spotipy.Spotify(auth=access_token)

    data = {}
    data["top_tracks_short"]  = sp.current_user_top_tracks(limit=5,  time_range="short_term")["items"]
    data["top_tracks_medium"] = sp.current_user_top_tracks(limit=10, time_range="medium_term")["items"]
    data["top_tracks_long"]   = sp.current_user_top_tracks(limit=10, time_range="long_term")["items"]
    data["top_artists_short"] = sp.current_user_top_artists(limit=5, time_range="short_term")["items"]
    recent = sp.current_user_recently_played(limit=10)["items"]
    data["recent_tracks"] = [item["track"] for item in recent if "track" in item]

    return data

def start_spotify_auth():
    sp_cfg = SpotifyConfig()
    sp_oauth = SpotifyOAuth(
        client_id=sp_cfg.client_id,
        client_secret=sp_cfg.client_secret,
        redirect_uri=sp_cfg.redirect_uri,
        scope=sp_cfg.scope,
        cache_path=".cache-streamlit",
        open_browser=False,
        show_dialog=True,
    )
    auth_url = sp_oauth.get_authorize_url()
    st.markdown(f"<meta http-equiv='refresh' content='0; url={auth_url}'>", unsafe_allow_html=True)
    st.stop()

def complete_spotify_auth_and_fetch_prefs():
    sp_cfg = SpotifyConfig()
    sp_oauth = SpotifyOAuth(
        client_id=sp_cfg.client_id,
        client_secret=sp_cfg.client_secret,
        redirect_uri=sp_cfg.redirect_uri,
        scope=sp_cfg.scope,
        cache_path=".cache-streamlit",
        open_browser=False,
        show_dialog=True,
    )

    token_info = sp_oauth.get_cached_token()
    if token_info and token_info.get("access_token"):
        if token_info.get("expires_at") and time.time() > token_info["expires_at"] - 30:
            token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        access_token = token_info["access_token"]
    else:
        code = get_query_param("code")
        if not code:
            raise RuntimeError("缺少 Spotify 回调参数 code。")
        raw = sp_oauth.get_access_token(code)
        access_token = raw["access_token"] if isinstance(raw, dict) else raw

    sp = spotipy.Spotify(auth=access_token)
    prefs = {}
    prefs["top_tracks_short"]  = sp.current_user_top_tracks(limit=5,  time_range="short_term")["items"]
    prefs["top_tracks_medium"] = sp.current_user_top_tracks(limit=10, time_range="medium_term")["items"]
    prefs["top_tracks_long"]   = sp.current_user_top_tracks(limit=10, time_range="long_term")["items"]
    prefs["top_artists_short"] = sp.current_user_top_artists(limit=5, time_range="short_term")["items"]
    recent = sp.current_user_recently_played(limit=10)["items"]
    prefs["recent_tracks"] = [it["track"] for it in recent if "track" in it]

    st.session_state.spotify_authed = True
    st.session_state.spotify_prefs = prefs
    st.session_state.spotify_error = None
    st.session_state.spotify_code_consumed = True
    clear_query_params()

# =========================
#         UI 组件
# =========================
def create_sidebar():
    """
    Create sidebar configuration interface
    """
    init_session_state()
    with st.sidebar:
        st.header("⚙️ System Configuration")

        stored_env_key = os.environ.get("OPENAI_API_KEY", "")
        default_key = st.session_state.get("openai_api_key", stored_env_key)

        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=default_key, 
            key="openai_api_key_input", 
            help="Please enter your OpenAI API key"
        )

        # 同步到 session_state
        if openai_api_key:
            st.session_state["openai_api_key"] = openai_api_key

        st.subheader("📁 Data Sources")
        config = RAGConfig()
        data_dir = Path(config.data_sources_path)

        if data_dir.exists():
            json_files = list(data_dir.glob("*.json"))
            jsonl_files = list(data_dir.glob("*.jsonl"))
            total_files = len(json_files) + len(jsonl_files)

            st.info(f"Constructing knowledge base from {total_files} clinical literature documents")
        else:
            st.warning(f"⚠️ Data directory not found: {config.data_sources_path}")
            st.info("Please create the directory and add your JSON/JSONL files")

        with st.expander("Playlist Settings", expanded=False):
            st.subheader("Spotify Authorization")

            code = get_query_param("code")
            if code and not st.session_state.get("spotify_code_consumed", False):
                with st.spinner("正在完成 Spotify 授权并获取你的音乐偏好…"):
                    try:
                        complete_spotify_auth_and_fetch_prefs()
                        st.rerun()
                    except Exception as e:
                        st.session_state.spotify_authed = False
                        st.session_state.spotify_prefs = None
                        st.session_state.spotify_error = str(e)

            if not st.session_state.get("spotify_authed", False) and not code:
                # 默认不强制；用户可手动点授权
                if st.button("Spotify Authorization", use_container_width=True):
                    with st.spinner("Redirecting to Spotify authorization…"):
                        start_spotify_auth()
                st.info("🎧 （可选）完成 Spotify 授权后，我会在“患者偏好策略”中参考你的常听艺人与歌曲")
                if st.session_state.get("spotify_error"):
                    st.warning(f"上次授权失败信息：{st.session_state['spotify_error']}")
            else:
                st.success("已完成 Spotify 授权并读取你的最近偏好")

            st.subheader("Playlist Generation Strategy")
            playlist_strategy = st.selectbox(
                "Strategy",
                ["Agent Generation based on literature recommendation", "Agent Tailored based on patient assessment"],
                index=0,
                help="选择歌单生成策略：默认按文献推荐，或基于患者偏好（偏好曲目≤50%）。"
            )

            playlist_size = st.slider(
                "Playlist Size",
                min_value=5,
                max_value=30,
                value=20,
                step=1,
                help="最终歌单曲目数量"
            )

            # 写入会话状态
            st.session_state["playlist_strategy"] = playlist_strategy
            st.session_state["playlist_size"] = playlist_size


        with st.expander("🔧 Advanced Settings", expanded=False):
            st.subheader("Model Settings")
            config.openai_model = st.selectbox(
                "OpenAI Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                index=0
            )

            config.openai_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Lower temperature makes answers more deterministic"
            )

            st.subheader("Retrieval Settings")
            config.chunk_size = st.number_input(
                "Document Chunk Size",
                min_value=256,
                max_value=4096,
                value=512,
                step=128,
                help="Number of characters per document chunk"
            )

            config.chunk_overlap = st.number_input(
                "Chunk Overlap Size",
                min_value=0,
                max_value=300,
                value=50,
                step=10,
                help="Number of overlapping characters between adjacent chunks"
            )

            config.similarity_top_k = st.number_input(
                "Number of Retrieved Documents",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of relevant documents retrieved per query"
            )

            config.similarity_cutoff = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Threshold for filtering low similarity documents"
            )

        return openai_api_key, config

def create_chat_interface():
    st.header("Chat with Music Therapy Expert")

    if 'rag_system' in st.session_state:
        stats = st.session_state.rag_system.get_system_stats()
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("Documents", stats["document_count"])
        with col2:
            st.metric("Data Files", stats["data_files_count"])
        with col3:
            st.metric("Retrieval Count", stats["config"]["similarity_top_k"])
        with col4:
            st.metric("Similarity Threshold", f"{stats['config']['similarity_cutoff']:.2f}")
        with col5:
            st.metric("Chroma Vectors", stats.get("chroma_vectors", 0))
        with col6:
            st.metric("Playlist Tracks", stats.get("playlist_tracks", 0))

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": INFO_MESSAGES["welcome"]}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def create_example_questions():
    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        col_idx = i % 2
        with cols[col_idx]:
            if st.button(question, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.spinner("Querying knowledge base..."):
                    try:
                        response = st.session_state.rag_system.query(question)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.rerun()
                    except Exception as e:
                        error_msg = ERROR_MESSAGES["query_error"].format(error=str(e))
                        st.error(error_msg)

def create_help_section():
    with st.expander("Help & Instructions", expanded=False):
        st.markdown("""
        ### How to Use
        1. **Configure API Key**: Enter your OpenAI API key in the sidebar
        2. **Check Data Sources**: Verify that JSON/JSONL files are available in the data directory
        3. **Initialize System**: Click "Initialize System" to build the knowledge base
        4. **Start Chatting**: Enter questions in the chat box or click example questions

        ### Data Format
        The system reads JSON/JSONL files with literature records containing:
        - **Title**: Research title
        - **Abstract**: Research abstract
        - **PMID**: PubMed identifier
        - **Authors**: Author information
        - **Journal**: Journal name
        - **Year**: Publication year
        - **Keywords**: Keywords

        ### Data Structure
        ```json
        {
            "title": "Music Therapy for Autism Spectrum Disorders",
            "abstract": "This study investigates...",
            "pmid": "12345678",
            "authors": "Smith, J. & Johnson, A.",
            "journal": "Journal of Music Therapy",
            "year": "2023"
        }
        ```
        """)

def main():
    config = RAGConfig()
    # 确保有歌单目录配置（如你的 config.py 暂无该字段，请加上：playlist_sources_path="playlist_docs"）
    if not hasattr(config, "playlist_sources_path"):
        config.playlist_sources_path = "playlist_docs"

    init_session_state()
    st.set_page_config(
        page_title=config.page_title,
        page_icon=config.page_icon,
        layout=config.layout,
    )

    st.title(f"Chat MusicTherapy")

    openai_api_key, user_config = create_sidebar()

    with st.sidebar:
        if st.button("🔄 Initialize System", use_container_width=True):
            openai_api_key = st.session_state.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
            if openai_api_key:
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    status_text.text("Initializing system...")
                    progress_bar.progress(20)

                    st.session_state.rag_system = MusicTherapyRAG(
                        openai_api_key=openai_api_key,
                        config=user_config
                    )

                    status_text.text("Loading documents...")
                    progress_bar.progress(40)

                    status_text.text("Building / Loading index...")
                    progress_bar.progress(60)

                    st.session_state.rag_system.initialize_system()

                    progress_bar.progress(100)
                    status_text.text("Initialization completed!")

                    st.success(SUCCESS_MESSAGES["system_ready"])
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(ERROR_MESSAGES["initialization_error"].format(error=str(e)))
            else:
                st.warning(ERROR_MESSAGES["no_api_key"])

    if 'rag_system' in st.session_state:
        st.success(SUCCESS_MESSAGES["system_initialized"])
        create_help_section()
        st.header("📖 Getting Started")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### Quick Start
            1. **Get API Key**: Visit [OpenAI website](https://openai.com/api/) to get an API key
            2. **Prepare Data**: Ensure JSON/JSONL files are in the `docs` directory
            3. **Add Playlists**: Put `song_*.json` files into the `playlist_docs` directory
            4. **Configure System**: Enter API key in the sidebar
            5. **Initialize**: Click "Initialize System" button
            6. **Start Chatting**: Enter questions in the chat interface

            ### Features
            - 🔍 **Literature RAG**: Hybrid retrieval (vector + keyword fallback)
            - 🎧 **Playlist Retrieval (No-Embed)**: In-memory structured filtering + preference rerank
            - 🤖 **Intent Routing**: literature vs playlist
            - ⚙️ **Flexible Configuration**: Adjustable system parameters
            """)

        with col2:
            st.markdown("""
            ### Data Format
            ```json
            {
              "title": "Study Title",
              "abstract": "Abstract text",
              "pmid": "12345678",
              "authors": "Authors",
              "journal": "Journal Name",
              "year": "2023"
            }
            ```

            ### Playlist Row (example)
            ```json
            {
              "No.": 1,
              "Song Title": "Clair de Lune",
              "Artist": "Debussy",
              "Album": "Piano Works",
              "Genre": ["Classical", "Instrumental"],
              "Duration": 292,
              "BPM": 72,
              "Lyrics": ""
            }
            ```
            """)

        tab1, tab2 = st.tabs(["💬 Smart Q&A", "💡 Example Questions"])
        with tab1:
            create_chat_interface()
        with tab2:
            create_example_questions()

        if prompt := st.chat_input("Please enter your question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Querying knowledge base..."):
                    try:
                        response = st.session_state.rag_system.query(prompt)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        error_msg = ERROR_MESSAGES["query_error"].format(error=str(e))
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.info(INFO_MESSAGES["system_instruction"])

        st.header("📖 Getting Started")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            ### Quick Start
            1. **Get API Key**: Visit [OpenAI website](https://openai.com/api/) to get an API key
            2. **Prepare Data**: Ensure JSON/JSONL files are in the `docs` directory
            3. **Add Playlists**: Put `song_*.json` files into the `playlist_docs` directory
            4. **Configure System**: Enter API key in the sidebar
            5. **Initialize**: Click "Initialize System" button
            6. **Start Chatting**: Enter questions in the chat interface

            ### Features
            - 🔍 **Literature RAG** (Chroma)
            - 🎧 **Playlist (No-Embed)**
            - 🧭 **Intent Router**
            """)
        with col2:
            st.markdown("""
            ### System Requirements
            - OpenAI API key
            - Spotify Authorization Token (if using Spotify)
            - Stable internet connection
            """)

if __name__ == "__main__":
    main()
