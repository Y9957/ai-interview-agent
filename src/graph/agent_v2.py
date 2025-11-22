# src/graph/agent_v2.py

import random
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# === 모듈 임포트 ===
from resume.resume_parser import analyze_resume
from strategy.strategy_generator import generate_question_strategy
from evaluation.evaluator import evaluate_answer, reflect, re_evaluate_answer
from generation.question_generator import (
    generate_question,
    summarize_interview,
    route_after_reflect,
    route_after_decide
)
from decision.decider import decide_next_step

# === 파일 텍스트 추출 ===
from utils.file_loader import extract_text_from_file    # 이 함수는 너가 만든 위치에 따라 경로 수정 필요


# ============================================================
# preProcessing_Interview 
# ============================================================
def preProcessing_Interview(file_path: str) -> Dict[str, Any]:
    # 파일 입력
    resume_text = extract_text_from_file(file_path)

    # state 초기화 
    initial_state: Dict[str, Any] = {
        "resume_text": resume_text,
        "resume_summary": "",
        "resume_keywords": [],
        "resume_sections": "",
        "question_strategy": {},
        "current_question": "",
        "current_answer": "",
        "current_strategy": "",
        "conversation": [],
        "evaluation": [],
        "next_step": "evaluate",
        "reflect_flag": False,
        "strategy_coverage": {},
        "used_questions": [],
        "need_re_eval": False,
        "decision": "generate",
    }

    # Resume 분석 (반드시 병합 반환)
    state = analyze_resume(initial_state)

    # 질문 전략 수립
    state = generate_question_strategy(state)

    # 첫 번째 질문 생성: '경력 및 경험'에서 1개 랜덤 선택
    example_questions = state["question_strategy"].get("경력 및 경험", {}).get("예시질문", [])
    selected_question = random.choice(example_questions) if example_questions else ""

    return {
        **state,
        "current_question": selected_question,
        "current_strategy": "경력 및 경험",
        "strategy_coverage": {**state.get("strategy_coverage", {}), "경력 및 경험": 1},
        "used_questions": state.get("used_questions", []) + ([selected_question] if selected_question else []),
        "next_step": "evaluate",
        "reflect_flag": False,
    }


# ============================================================
# update_current_answer 
# ============================================================
def update_current_answer(state: Dict[str, Any], answer: str) -> Dict[str, Any]:
    return {**state, "current_answer": answer}


# ============================================================
# LangGraph 구성 
# ============================================================

# StateGraph 정의
builder = StateGraph(dict)

builder.add_node("evaluate",    evaluate_answer)
builder.add_node("reflect",     reflect)
builder.add_node("re_evaluate", re_evaluate_answer)
builder.add_node("decide",      decide_next_step)
builder.add_node("generate",    generate_question)
builder.add_node("summarize",   summarize_interview)

# entry point
builder.set_entry_point("evaluate")

# evaluate → reflect
builder.add_edge("evaluate", "reflect")

# reflect → re_evaluate / decide
builder.add_conditional_edges(
    "reflect",
    route_after_reflect,
    {"re_evaluate": "re_evaluate", "decide": "decide"}
)

# re_evaluate → decide
builder.add_edge("re_evaluate", "decide")

# decide → generate / summarize
builder.add_conditional_edges(
    "decide",
    route_after_decide,
    {"generate": "generate", "summarize": "summarize"}
)

# generate, summarize → END
builder.add_edge("generate", END)
builder.add_edge("summarize", END)

# compile
graph = builder.compile()

