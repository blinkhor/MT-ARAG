# 🎵 音乐治疗知识库 RAG 系统

基于 LlamaIndex 构建的音乐治疗文献查询系统，支持从 Excel 文件读取文献数据，并提供智能问答功能。

## ✨ 功能特点

- 📊 **Excel 数据导入**：支持从 Excel 文件读取音乐治疗文献数据
- 🤖 **智能问答**：基于 RAG 技术的专业音乐治疗问答
- 🎯 **语义检索**：使用向量数据库进行高效的语义搜索
- 💬 **友好界面**：基于 Streamlit 的交互式 Web 界面
- 📚 **知识库管理**：持久化存储，支持增量更新

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd music-therapy-rag

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

#### 获取 OpenAI API 密钥
1. 访问 [OpenAI 官网](https://openai.com/api/)
2. 注册并获取 API 密钥
3. 确保账户有足够的额度

#### 准备 Excel 数据文件
确保您的 Excel 文件包含以下列（建议格式）：

| 列名 | 描述 | 示例 |
|------|------|------|
| Title | 研究标题 | "Music Therapy for Autism Spectrum Disorder" |
| Author | 作者 | "Smith, J. & Johnson, A." |
| Year | 发表年份 | 2023 |
| Journal | 期刊名称 | "Journal of Music Therapy" |
| Abstract | 摘要 | "This study investigates..." |
| Keywords | 关键词 | "autism, music therapy, intervention" |
| DOI | 文献DOI | "10.1093/jmt/thxx001" |
| Methods | 研究方法 | "Randomized controlled trial" |
| Results | 研究结果 | "Significant improvement observed" |
| Conclusion | 结论 | "Music therapy shows promise..." |

### 3. 运行系统

```bash
streamlit run main.py
```

系统将在浏览器中打开（通常是 http://localhost:8501）

### 4. 使用步骤

1. **配置系统**
   - 在侧边栏输入 OpenAI API 密钥
   - 上传音乐治疗文献 Excel 文件

2. **初始化知识库**
   - 点击"初始化系统"按钮
   - 等待系统构建向量索引（首次运行可能需要几分钟）

3. **开始对话**
   - 在聊天界面输入问题
   - 系统会基于文献数据提供专业回答

## 📋 Excel 文件格式要求

### 基本要求
- 文件格式：`.xlsx` 或 `.xls`
- 第一行必须是列标题
- 每行代表一篇文献

### 推荐列结构
```
Title | Author | Year | Journal | Abstract | Keywords | DOI | Methods | Results | Conclusion
```

### 数据质量建议
- **完整性**：尽量填写所有重要字段
- **一致性**：保持格式统一（如年份格式、作者名格式）
- **准确性**：确保数据的准确性和可靠性

## 🔧 高级配置

### 环境变量设置
创建 `.env` 文件：

```env
OPENAI_API_KEY=your_api_key_here
CHUNK_SIZE=512
CHUNK_OVERLAP=50
SIMILARITY_TOP_K=5
SIMILARITY_CUTOFF=0.7
```

### 自定义配置
修改 `main.py` 中的参数：

```python
# 文档分块设置
node_parser = SimpleNodeParser.from_defaults(
    chunk_size=512,      # 调整块大小
    chunk_overlap=50     # 调整重叠大小
)

# 检索设置
retriever = VectorIndexRetriever(
    index=self.index,
    similarity_top_k=5   # 调整检索数量
)

# 相似度阈值
postprocessor = SimilarityPostprocessor(
    similarity_cutoff=0.7  # 调整相似度阈值
)
```

## 📚 使用示例

### 示例问题
- "音乐治疗对自闭症儿童有什么帮助？"
- "音乐治疗在老年痴呆症治疗中的应用效果如何？"
- "即兴音乐治疗的主要技术有哪些？"
- "音乐治疗师需要具备什么资质？"

### 查询技巧
1. **具体化问题**：越具体的问题，答案越准确
2. **使用专业术语**：使用音乐治疗领域的专业词汇
3. **多角度提问**：从不同角度探索同一主题

## 🛠️ 故障排除



## 📦 项目结构

```
music-therapy-rag/
├── main.py                 # 主程序文件
├── requirements.txt        # 依赖包列表
├── README.md              # 项目说明
├── .env                   # 环境变量（需自创建）
├── chroma_db/             # 向量数据库存储目录
└── temp_excel.xlsx        # 临时Excel文件
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE) 文件。

## 🆘 支持

如果您遇到问题或有任何建议，请：

1. 查看故障排除部分
2. 搜索已有的 Issues
3. 创建新的 Issue 并详细描述问题

## 🎯 路线图

- [ ] 支持更多文件格式（CSV、JSON）
- [ ] 添加文献去重功能
- [ ] 支持多语言查询
- [ ] 添加可视化分析功能
- [ ] 集成更多向量数据库选项
- [ ] 添加用户管理功能

---

**注意**：本系统仅供学术研究和教育目的使用。使用时请确保遵守相关的版权和使用条款。