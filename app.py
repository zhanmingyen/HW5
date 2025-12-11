import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

# 這個模型是 OpenAI 釋出的 GPT-2 偵測器
# id2label: 0 -> "Fake" (比較像 GPT-2 產生), 1 -> "Real" (比較像人類撰寫)
MODEL_NAME = "openai-community/roberta-base-openai-detector"


@st.cache_resource
def load_model():
    """只在第一次呼叫時下載 / 載入模型，之後重用同一份權重。"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def predict_proba(text: str, tokenizer, model):
    """對一段文字做推論，回傳 (ai_prob, human_prob)。"""
    # 將文字編碼成張量（最多 512 token 就好）
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]  # shape: [2]

    probs = torch.softmax(logits, dim=-1).tolist()
    # 根據 config.json: 0 = "Fake" (AI-like), 1 = "Real" (Human-like)
    ai_prob = probs[0]
    human_prob = probs[1]
    return ai_prob, human_prob


def main():
    st.set_page_config(
        page_title="AI / Human 文章偵測器",
        page_icon="🤖",
        layout="centered",
    )

    st.title("🤖 AI / Human 文章偵測器 (Demo)")
    st.write(
        """
        輸入一段文字，模型會估計這段文字比較像是 **AI 生成** 還是 **人類撰寫**，
        並給出對應的機率百分比（AI% / Human%）。
        """
    )
    st.caption(
        "⚠️ 本工具僅供課程 / 研究 **教學示範**，"
        "準確率有限，請勿作為學術違規或抄襲判定的唯一依據。"
    )

    with st.sidebar:
        st.header("設定")
        st.markdown(
            f"**偵測模型：** `{MODEL_NAME}`  \n"
            "這是 RoBERTa base 模型微調而成的 GPT-2 文字偵測器。"
        )
        threshold = st.slider(
            "判定為「AI 生成」的門檻 (AI%)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
        )
        st.caption(
            "例如門檻 = 0.7，代表 AI 機率 ≥ 70% 時，就顯示為「較像 AI」。"
        )

    st.subheader("輸入待檢測文字")
    text = st.text_area(
        label="請在這裡貼上要偵測的文章（中英文皆可）：",
        height=200,
        placeholder="例如：本研究旨在探討......",
    )

    # 載入模型（第一次會下載 & load，需要一點時間）
    tokenizer, model = load_model()

    if text.strip():
        # 做推論
        ai_prob, human_prob = predict_proba(text, tokenizer, model)
        ai_pct = ai_prob * 100
        human_pct = human_prob * 100

        # 文字判斷結果
        st.subheader("判斷結果")

        if ai_prob >= threshold:
            label_text = "看起來 **較像 AI 產生的內容** 🤖"
        else:
            label_text = "看起來 **較像人類撰寫的內容** 🧑"

        st.markdown(
            f"""
            ### {label_text}

            - **AI 機率 (Fake / Model-Generated)**：`{ai_pct:.2f}%`  
            - **Human 機率 (Real / Human-Written)**：`{human_pct:.2f}%`
            """
        )

        # 簡單視覺化：長條圖顯示 AI% / Human%
        st.subheader("機率分佈 (AI% vs Human%)")
        df = pd.DataFrame(
            {
                "類別": ["AI 生成", "Human"],
                "機率百分比": [ai_pct, human_pct],
            }
        )
        st.bar_chart(df.set_index("類別"))

        # 額外資訊：一些簡單統計量（可選）
        st.subheader("文字統計 (選用)")
        num_chars = len(text)
        num_words = len(text.split())
        num_lines = len(text.splitlines())

        col1, col2, col3 = st.columns(3)
        col1.metric("字元數 (characters)", num_chars)
        col2.metric("詞數 (words, 以空白切)", num_words)
        col3.metric("行數 (lines)", num_lines)

        with st.expander("模型技術說明 / 限制（可以寫在報告說明）"):
            st.markdown(
                """
                - 使用的模型：`openai-community/roberta-base-openai-detector`  
                  - 這個模型是 RoBERTa base 微調而成，用來分辨文字是否由 GPT-2 產生。:contentReference[oaicite:1]{index=1}  
                - 模型輸出兩個類別：  
                  - `Fake`：較可能是 **模型產生 (AI-like)**  
                  - `Real`：較可能是 **人類撰寫 (Human-like)**  
                - 我們將 `Fake` 視為 AI 生成機率 (AI%)，`Real` 視為 Human 機率 (Human%)。  
                - 由於訓練資料主要來自 GPT-2 的輸出，對 ChatGPT / GPT-4 等較新模型，
                  偵測效果有限，只能做**傾向性判斷**，不是絕對事實。:contentReference[oaicite:2]{index=2}
                """
            )
    else:
        st.info("👆 請先在上方輸入一段文字，系統才會進行 AI / Human 判斷。")


if __name__ == "__main__":
    main()
