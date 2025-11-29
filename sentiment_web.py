
import streamlit as st
from transformers import pipeline
import plotly.express as px

# 1. 載入更準確的中文情感模型
@st.cache_resource(show_spinner=False)
def load_model():
    # 單純 positive / negative 的中文模型，準確度高且速度快
    return pipeline(
        "sentiment-analysis",
        model="uer/roberta-base-finetuned-jd-binary-chinese"
    )

classifier = load_model()

# 2. 介面
st.title("中文情感分析小工具")
st.markdown("貼上一段中文文字，我來分析你這段話的情感喔！")

text = st.text_area("輸入中文文本：", height=150)

# 3. 按鈕
if st.button("分析情感"):
    if not text.strip():
        st.warning("請輸入文字喔！")
    else:
        with st.spinner("分析中..."):
            result = classifier(text)[0]

            # 抓輸出
            label = result["label"].lower()
            score = result["score"]

            # 保護 progress bar（避免某些模型輸出 logits 而報錯）
            score_safe = min(max(score, 0), 1)

            # 4. label mapping（適用多數模型）
            if "pos" in label:
                sentiment = "正面 😊"
                color = "green"
            elif "neg" in label:
                sentiment = "負面 😡"
                color = "red"
            else:
                sentiment = "中性 😐"
                color = "gray"

            # 5. 顯示結果
            st.markdown(f"### 情感傾向：**{sentiment}**")
            st.write(f"信心分數：{score:.3f}")
            st.progress(score_safe)

            # Plotly 圖表（修正顏色）
            fig = px.pie(
                values=[score_safe, 1 - score_safe],
                names=[sentiment, "其他"],
                color_discrete_sequence=[color, "lightgray"]
            )
            st.plotly_chart(fig)
