import os
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# ✅ API 키 설정
openai.api_key = st.secrets["OPENAI_API_KEY"]
api_key = st.secrets["OPENAI_API_KEY"]

# ✅ 제목과 설명
st.title("📄 청년공약 기반 정책 챗봇")
st.markdown("💬 PDF 내용을 기반으로 GPT가 답변해드립니다.")

# ✅ PDF 로딩 및 처리
loader = PyPDFLoader("청년공약.pdf")
pages = loader.load()

# ✅ 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(pages)

# ✅ 벡터 저장소 생성
embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
db = FAISS.from_documents(texts, embeddings)

# ✅ 질의응답 체인 구성
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0, openai_api_key=api_key),
    chain_type="stuff",
    retriever=db.as_retriever()
)

# ✅ 질문 입력받기
question = st.text_input("✍️ 궁금한 정책 내용을 입력하세요:")

if question:
    with st.spinner("GPT가 정책자료를 읽고 답변 중..."):
        answer = qa.run(question)
        st.success(answer)
else:
    st.warning("질문을 입력해 주세요")
