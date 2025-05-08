
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import openai

# ✅ 제목과 설명
st.title("📄 청년공약 기반 정책 챗봇")
st.markdown("💬 PDF 내용을 기반으로 GPT가 답변해드립니다.")

# ✅ API 키 입력 (비밀번호 형식으로 숨김)
<<<<<<< HEAD
openai.api_key = st.secrets["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"]
=======
openai.api_key = st.secrets["OPENAI_API_KEY"]

>>>>>>> 71a2419 (add secrets.toml and initial deploy setup)


    # ✅ PDF 로딩 및 처리
loader = PyPDFLoader("청년공약.pdf")
pages = loader.load()

    # ✅ 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(pages)

    # ✅ 벡터 저장소 생성
embeddings = OpenAIEmbeddings(openai_api_key="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")
db = FAISS.from_documents(texts, embeddings)

    # ✅ 질의응답 체인 구성
qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, openai_api_key="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"),
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    # ✅ 일반 텍스트 입력창으로 질문 받기
question = st.text_input("✍️ 궁금한 정책 내용을 입력하세요:")

if question:
        with st.spinner("GPT가 정책자료를 읽고 답변 중..."):
            answer = qa.run(question)
            st.success(answer)
else:
    st.warning("질문을 입력해 주세요")
