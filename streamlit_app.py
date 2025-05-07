import streamlit as st
from openai import OpenAI

# OpenAI API 키
client = OpenAI(api_key="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")

st.title("나만의 GPT 챗봇")
st.markdown("💬 질문을 입력해보세요 (GPT 연동 완료)")

# 👉 여기서 user_input을 먼저 정의
user_input = st.text_input("You:")

if user_input:
    with st.spinner("GPT가 응답 생성 중..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 유쾌하고 똑똑한 한국 정치 전문가야."},
                {"role": "user", "content": user_input}
            ]
        )
        bot_reply = response.choices[0].message.content
        st.success(bot_reply)
