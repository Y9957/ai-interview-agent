# src/evaluation/evaluator.py

import ast
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Any


# ==============================
# evaluate_answer
# ==============================
def evaluate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    현재 질문/답변을 두 항목(질문과의 연관성, 답변의 구체성)으로 평가하고
    conversation/evaluation을 갱신한 뒤 다음 스텝을 'reflect'로 설정한다.
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # --- 입력 값 추출 ---
    current_question  = state.get("current_question", "")
    current_answer    = state.get("current_answer", "")
    current_strategy  = state.get("current_strategy", "")
    question_strategy = state.get("question_strategy", {})
    resume_summary    = state.get("resume_summary", "")
    resume_keywords   = ", ".join(state.get("resume_keywords", []))

    # --- 질문전략 블록 추출(유연 처리) ---
    strategy_block = ""
    if isinstance(question_strategy, dict):
        strategy_block = question_strategy.get(current_strategy, {}).get("질문전략", "")
    elif isinstance(question_strategy, str):
        try:
            parsed = ast.literal_eval(question_strategy)
            strategy_block = parsed.get(current_strategy, {}).get("질문전략", "")
        except Exception:
            strategy_block = ""

    # --- 프롬프트 구성 ---
    prompt = ChatPromptTemplate.from_template("""
당신은 인터뷰 평가를 위한 AI 평가자입니다.
[참고 정보]
- 이력서 요약: {resume_summary}
- 이력서 키워드: {resume_keywords}
- 질문 전략({current_strategy}): {strategy}
- 질문: {question}
- 답변: {answer}

아래 두 항목을 '상/중/하'로만 평가하고, 딕셔너리 literal 하나만 출력하세요.
{{
  "질문과의 연관성": "<상/중/하>",
  "답변의 구체성": "<상/중/하>"
}}
""")

    formatted = prompt.format(
        resume_summary=resume_summary,
        resume_keywords=resume_keywords,
        strategy=strategy_block,
        current_strategy=current_strategy,
        question=current_question,
        answer=current_answer,
    )

    # --- LLM 호출 & 결과 파싱(안전) ---
    raw = llm.invoke(formatted).content.strip()
    try:
        eval_result = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except Exception:
        eval_result = {"질문과의 연관성": "중", "답변의 구체성": "중"}

    # --- 짧은 답변 자동 하향 ---
    if not current_answer or len(current_answer.strip()) < 20:
        eval_result["질문과의 연관성"] = "하"
        eval_result["답변의 구체성"] = "하"

    # --- conversation 갱신(중복 최소화) ---
    conversation = list(state.get("conversation", []))
    if not conversation or not (
        conversation[-1].get("question") == current_question and
        conversation[-1].get("answer") == current_answer
    ):
        conversation.append({
            "question": current_question,
            "answer": current_answer,
            "strategy": state.get("current_strategy", "")
        })

    # --- evaluation 갱신 ---
    evaluation = list(state.get("evaluation", []))
    eval_result["question_index"] = len(conversation) - 1
    evaluation.append(eval_result)

    # --- 상태 반환 ---
    return {
        **state,
        "conversation": conversation,
        "evaluation": evaluation,
        "next_step": "reflect",
    }


# ==============================
# reflect 
# ==============================
def reflect(state: Dict[str, Any]) -> Dict[str, Any]:
    # 재평가 직후 한 번은 통과(무한 re_evaluate 방지)
    if state.get("reflect_flag", False):
        return {
            "reflection_status": "정상",
            "need_re_eval": False,
            "reflect_flag": False,
            "next_step": "decide",
        }

    eval_list = state.get("evaluation", [])
    if not eval_list:
        return {
            "reflection_status": "정상",
            "need_re_eval": False,
            "next_step": "decide",
        }

    last = eval_list[-1]
    rel = last.get("질문과의 연관성", "중")
    spc = last.get("답변의 구체성", "중")
    ans = (state.get("current_answer") or "").strip()

    too_short  = len(ans) < 40
    has_number = bool(re.search(r"\d", ans) or re.search(r"%", ans))
    has_detail = bool(re.search(r"(수치|기간|개월|년|지표|정확도|MAE|RMSE|건|명)", ans))

    need_re, reason = False, ""

    if too_short:
        need_re, reason = True, "답변 길이 부족"
    elif rel == "상" and spc == "상" and (not has_number and not has_detail):
        need_re, reason = True, "과관대: 근거/수치 없음"
    elif rel == "하" and spc == "상":
        need_re, reason = True, "평가 모순(연관성 하·구체성 상)"
    elif rel == "하" and spc == "하" and len(ans) > 180 and (has_number or has_detail):
        need_re, reason = True, "과엄격: 근거 충분"

    if need_re:
        return {
            "reflection_status": "재평가 필요",
            "need_re_eval": True,
            "reflect_flag": True,
            "reflection_reason": reason,
            "next_step": "re_evaluate",
        }

    return {
        "reflection_status": "정상",
        "need_re_eval": False,
        "reflect_flag": False,
        "reflection_reason": "평가 수용",
        "next_step": "decide",
    }


# ==============================
# re_evaluate_answer 
# ==============================
def re_evaluate_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
당신은 인터뷰 평가자입니다. 아래 질문-답변을 엄격한 기준으로 다시 평가하세요.
규칙:
- 답변이 짧거나(40자 미만) 근거(수치, 기간, 지표)가 없으면 낮게 평가.
- 평가 항목: "질문과의 연관성", "답변의 구체성".
- 값: '상'/'중'/'하'.
- 출력은 딕셔너리 literal 한 개만.

[질문]
{question}

[답변]
{answer}
""")

    formatted = prompt.format(
        question=state.get("current_question", ""),
        answer=state.get("current_answer", "")
    )

    raw = llm.invoke(formatted).content.strip()

    try:
        new_eval = ast.literal_eval(raw)
    except Exception:
        new_eval = {"질문과의 연관성": "중", "답변의 구체성": "중"}

    prev_evals = list(state.get("evaluation", []))
    q_idx = (prev_evals[-1] if prev_evals else {}).get(
        "question_index",
        max(0, len(state.get("conversation", [])) - 1)
    )
    new_eval["question_index"] = q_idx

    if prev_evals:
        prev_evals[-1] = new_eval
    else:
        prev_evals = [new_eval]

    return {
        "evaluation": prev_evals,
        "reflect_flag": False,
        "need_re_eval": False,
        "reflection_status": "정상",
        "next_step": "decide",
    }

