import os
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

AI_PROVIDERS = {
    "免费网关 (无需Key)": {
        "base_url": "https://opencode.ai/zen/v1",
        "api_key": "public",
        "model": "mimo-v2.5-free"
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "api_key": "",
        "model": "deepseek-chat"
    },
    "Kimi (月之暗面)": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key": "",
        "model": "moonshot-v1-8k"
    },
    "智谱 GLM": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": "",
        "model": "glm-4-flash"
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        "model": "gpt-4o-mini"
    }
}

st.set_page_config(page_title="RAG知识库检索", layout="wide")
st.title("RAG知识库检索")
st.markdown("请先在左侧上传文档并加载，然后在右侧选择 AI 并提问。")

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None
if "texts" not in st.session_state:
    st.session_state.texts = []

def load_document(file_path):
    """加载文档、切分、建立 TF-IDF 索引"""
    file_name = os.path.basename(file_path)
    
    if file_name.endswith(".txt"):
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        return "不支持的文件类型，请选择 txt、pdf 或 docx 文件。"
    
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
        chunks = splitter.split_documents(loader.load())
        texts = [chunk.page_content for chunk in chunks]
        
        vectorizer = TfidfVectorizer().fit(texts)
        tfidf_matrix = vectorizer.transform(texts)
        
        # 保存到 session_state
        st.session_state.vectorizer = vectorizer
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.texts = texts
        
        return f"成功加载 {file_name}，共 {len(texts)} 块文本。现在可以提问了！"
    except Exception as e:
        return f"加载失败：{str(e)}"

def ask_question(question, provider_name, user_api_key):
    """检索 + 调用大模型生成答案"""
    if st.session_state.vectorizer is None or st.session_state.tfidf_matrix is None:
        return "请先加载文档。"
    
    if question.strip() == "":
        return "问题不能为空。"
    
    # 检索最相似的文本块
    question_vector = st.session_state.vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, st.session_state.tfidf_matrix)
    most_similar_index = np.argmax(similarities)
    most_similar_chunk = st.session_state.texts[most_similar_index]
    
    final_prompt = f"Content: {most_similar_chunk}\nQuestion: {question}\nAnswer:"
    
    config = AI_PROVIDERS[provider_name]
    api_key = user_api_key.strip() if user_api_key and user_api_key.strip() else config["api_key"]
    
    if not api_key and provider_name != "免费网关 (无需Key)":
        return f"你选择了 {provider_name}，请填入该平台的 API Key。"
    
    try:
        llm = ChatOpenAI(
            model=config["model"],
            api_key=api_key,
            base_url=config["base_url"],
            temperature=0.2,
        )
        response = llm.invoke(final_prompt)
        return response.content
    except Exception as e:
        return f"{provider_name} 调用失败：{str(e)}"

def clear_document():
    """清空当前文档索引"""
    st.session_state.vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.texts = []

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("文档管理")
    
    uploaded_file = st.file_uploader("上传文档", type=["txt", "pdf", "docx"])
    
    if st.button("加载文档", type="primary"):
        if uploaded_file is None:
            st.warning("请先上传一个文档。")
        else:
            # 把上传的文件保存到临时路径
            temp_path = os.path.join("./data", uploaded_file.name)
            os.makedirs("./data", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            status = load_document(temp_path)
            st.success(status)
    
    if st.button("删除当前文档", type="secondary"):
        clear_document()
        st.info("已清空当前文档，请重新上传。")


with col2:
    st.subheader("AI 问答")
    
    provider_name = st.selectbox(
        "选择 AI 提供商",
        options=list(AI_PROVIDERS.keys()),
        index=0
    )
    
    user_api_key = st.text_input(
        "API Key（如果选了免费网关，这里可以留空）",
        type="password",
        placeholder="sk-..."
    )
    
    question = st.text_input("请输入你的问题")
    
    if st.button("提问", type="primary"):
        if not question:
            st.warning("请输入问题。")
        else:
            with st.spinner("思考中..."):
                answer = ask_question(question, provider_name, user_api_key)
            st.markdown("### AI 的回答")
            st.write(answer)