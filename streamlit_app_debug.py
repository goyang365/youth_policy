
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ✅ 디버그 모드 여부 (True면 실제 GPT 호출, False면 모의 응답)
USE_OPENAI = False

st.title("📄 청년공약 기반 정책 챗봇 (디버그 모드)")
st.markdown("💬 PDF 내용을 기반으로 GPT가 답변해드립니다. (API 없이 테스트)")

# ✅ PDF 로딩 및 처리
from pathlib import Path
pdf_path = Path(__file__).parent / "청년공약.pdf"
loader = PyPDFLoader(str(pdf_path))
pages = loader.load()  # ✅ 이 줄 꼭 있어야 해!

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(pages)


if USE_OPENAI:
    import openai
    from langchain.chat_models import ChatOpenAI
    from langchain.embeddings import OpenAIEmbeddings

    openai.api_key = st.secrets["OPENAI_API_KEY"]
    api_key = st.secrets["OPENAI_API_KEY"]

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, openai_api_key=api_key),
        chain_type="stuff",
        retriever=db.as_retriever()
    )
else:
    qa = None

# ✅ 질문 입력
question = st.text_input("✍️ 궁금한 정책 내용을 입력하세요:")

if question:
    with st.spinner("GPT가 정책자료를 읽고 답변 중..."):
        if USE_OPENAI and qa:
            answer = qa.run(question)
        else:
            answer = f"💡 [모의 응답] '{question}' 에 대한 답변입니다. 실제 GPT 없이 테스트 중입니다."
        st.success(answer)
else:
    st.info("테스트용 질문을 입력해 보세요.")
