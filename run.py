# run.py
import os
from src.graph.agent_v2 import preProcessing_Interview, update_current_answer, graph

def main():
    print("=== AI Interview Agent (CLI 모드) ===")
    
    file_path = input("이력서 파일 경로(PDF 또는 DOCX)를 입력하세요: ").strip()
    if not os.path.exists(file_path):
        print("❌ 파일을 찾을 수 없습니다.")
        return

    # 초기 상태 생성
    state = preProcessing_Interview(file_path)
    print("\n[AI 면접관]:", state["current_question"])
    
    # 인터뷰 루프
    while True:
        user_answer = input("\n[지원자]: ").strip()
        state = update_current_answer(state, user_answer)

        # LangGraph 실행
        state = graph.invoke(state)

        # 종료 판정
        if state.get("next_step") == "end":
            print("\n=== 인터뷰 종료 ===")
            print("\n📋 [최종 면접 보고서]")
            print(state.get("summary_report", "⚠ 보고서 생성 실패"))

            again = input("\n인터뷰를 다시 진행할까요? (예/아니오): ").strip().lower()
            if again in ["예", "yes", "y"]:
                # 초기화
                state = preProcessing_Interview(file_path)
                print("\n[AI 면접관]:", state["current_question"])
                continue
            else:
                print("면접이 종료되었습니다.")
                break
        
        # 다음 질문 출력
        print("\n[AI 면접관]:", state["current_question"])


if __name__ == "__main__":
    main()
