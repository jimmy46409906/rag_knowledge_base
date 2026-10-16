# 📚 轻量级本地 RAG 知识库

基于 LangChain + Streamlit + 多模型 API 构建的本地 RAG 知识库问答系统。

## ✨ 功能特性

- 支持上传 TXT / PDF / DOCX 文档
- 自动切分文档并建立本地 TF-IDF 检索索引
- 检索增强生成（RAG），基于私有资料精准回答，有效缓解大模型幻觉
- 支持免费网关、DeepSeek、Kimi、智谱 GLM、OpenAI 多模型动态切换
- 用户可自行填入 API Key，无需硬编码

## 🛠️ 技术栈

- **框架**：LangChain、Streamlit
- **检索**：Scikit-learn（TF-IDF + 余弦相似度）
- **大模型**：DeepSeek / Kimi / 智谱 GLM / OpenAI / 免费网关
- **文档解析**：pypdf、docx2txt

## 🚀 快速开始

### 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py