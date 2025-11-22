# app.py
import gradio as gr
from src.graph.agent_v2 import (
    preProcessing_Interview,
    update_current_answer,
    graph
)

# 세션 상태 초기화
def init_state():
    return {
        "state": None,
        "started": False,
        "ended": False,
        "history": []
    }

# 파일 업로드 & 준비
def upload_resume(file_obj, session_state):
    if file_obj is None:
        return session_state, "❗ 이력서를 업로드해주세요."

    file_path = file_obj.name
    state = preProcessing_Interview(file_path)

    session_state["state"] = state
    session_state["started"] = True
    session_state["history"] = [["🤖 AI 면접관", state["current_question"]]]

    return session_state, session_state["history"]

# 답변 처리
def chat(user_text, session_state):
    if not session_state["started"]:
        return session_state, [["❗ 먼저 이력서를 업로드 해주세요."]]

    if session_state["ended"]:
        # 재시작 여부
        if user_text.strip().lower() in ["예", "yes", "y"]:
            old = session_state["state"]
            new_state = preProcessing_Interview(old.get("resume_text_path", ""))
            session_state["state"] = new_state
            session_state["ended"] = False
            session_state["history"] = [["🤖 AI 면접관", new_state["current_question"]]]
            return session_state, session_state["history"]
        else:
            session_state["history"].append(["🤖 AI 면접관", "면접을 종료합니다."])
            return session_state, session_state["history"]

    # 일반 답변 처리
    session_state["history"].append(["🙋 지원자", user_text])
    session_state["state"] = update_current_answer(session_state["state"], user_text)

    # LangGraph 실행
    session_state["state"] = graph.invoke(session_state["state"])

    # 종료 여부
    if session_state["state"]["next_step"] == "end":
        session_state["ended"] = True

        report = session_state["state"].get("summary_report", "")
        session_state["history"].append(["📋 면접 보고서", report])
        session_state["history"].append(["🤖 AI 면접관", "인터뷰가 종료되었습니다. 다시 진행할까요? (예/아니오)"])

        return session_state, session_state["history"]

    # 다음 질문
    next_q = session_state["state"]["current_question"]
    session_state["history"].append(["🤖 AI 면접관", next_q])

    return session_state, session_state["history"]

# UI 구성
with gr.Blocks() as demo:
    session = gr.State(init_state())

    gr.Markdown("# 🤖 AI Interview Agent\n이력서를 업로드하고 면접을 시작하세요!")

    with gr.Row():
        file_input = gr.File(label="📄 이력서 업로드 (PDF 또는 DOCX)")
        start_btn = gr.Button("인터뷰 시작")

    chatbox = gr.Chatbot(height=500)
    textbox = gr.Textbox(placeholder="답변을 입력하고 Enter를 누르세요.", show_label=False)

    start_btn.click(upload_resume, inputs=[file_input, session], outputs=[session, chatbox])
    textbox.submit(chat, inputs=[textbox, session], outputs=[session, chatbox])
    textbox.submit(lambda: "", None, textbox)

demo.launch()
