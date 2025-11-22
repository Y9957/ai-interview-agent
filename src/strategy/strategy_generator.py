# src/strategy/strategy_generator.py

import ast
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict
from typing import Any

def generate_question_strategy(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    질문 전략 생성 함수 
    """

    resume_summary = state.get("resume_summary", "")
    resume_keywords = ", ".join(state.get("resume_keywords", []))

    prompt = ChatPromptTemplate.from_template("""
당신은 전문 AI 면접관입니다.
아래 이력서 요약과 키워드를 분석하여 지원자의 강점과 보완점을 먼저 파악한 뒤,
5가지 면접 질문 부문별로 질문 전략과 예시 질문을 작성하세요.

- 이력서 요약:
{resume_summary}

- 핵심 키워드:
{resume_keywords}

[출력형식]
딕셔너리 형태로 작성:
{
"경력 및 경험": {
    "질문전략": "지원자의 주요 프로젝트와 기술 경험을 중심으로 실무 이해도를 평가합니다.",
    "예시질문": [
        "프로젝트 수행 시 가장 도전적이었던 기술적 문제는 무엇이었습니까?",
        "협업 과정에서 본인이 맡은 역할과 팀 내 기여도를 설명해주세요."
    ]
}},
"동기 및 커뮤니케이션": {
    "질문전략": "...",
    "예시질문": ["...", "..."]
}},
"논리적 사고": {
    "질문전략": "...",
    "예시질문": ["...", "..."]
}},
"기술 역량 및 전문성": {
    "질문전략": "...",
    "예시질문": ["...", "..."]
}},
"성장 가능성 및 자기주도성": {
    "질문전략": "...",
    "예시질문": ["...", "..."]
}}
}
""")

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.4)

    formatted = prompt.format(
        resume_summary=resume_summary,
        resume_keywords=resume_keywords
    )

    response = llm.invoke(formatted)
    raw_text = response.content.strip()

    # 딕셔너리 변환 
    try:
        strategy_dict = ast.literal_eval(raw_text)
    except Exception as e:
        raise ValueError("question_strategy를 딕셔너리로 변환하는 데 실패했습니다.\n원본:\n" + raw_text) from e

    return {
        **state,
        "question_strategy": strategy_dict
    }
