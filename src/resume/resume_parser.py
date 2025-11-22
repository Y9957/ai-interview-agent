# src/resume/resume_parser.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser

def analyze_resume(state):
    """
    이력서 분석 전체 함수 (요약 + 섹션 + 키워드 추출)
    """
    resume_text = state.get("resume_text", "")
    if not resume_text:
        raise ValueError("resume_text가 비어 있습니다. 먼저 텍스트를 추출해야 합니다.")

    # llm 준비
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    # (1) 전체 요약
    summary_prompt = ChatPromptTemplate.from_template(
        """당신은 이력서를 바탕으로 인터뷰 질문을 설계하는 AI입니다.
        다음 이력서 및 자기소개서 내용에서 질문을 뽑기 위한 중요한 내용을 10문장 정도로 요약을 해줘
        (요약시 ** 기호는 사용하지 말것)
- 프로젝트, 경험, 기술, 자격증, 동기 등이 드러나게 써라.

본문:
{resume_text}
"""
    )
    summary_msg = summary_prompt.format(resume_text=resume_text)
    summary_resp = llm.invoke(summary_msg)
    resume_summary = summary_resp.content.strip()

    # (2) 섹션 분리 요약
    section_prompt = ChatPromptTemplate.from_template(
        """
당신은 아래 이력서를 분석해서 중요한 정보를 5개 섹션으로 나누어 정리합니다.

=== 직무/관심 ===
내용
=== 프로젝트/활동 ===
내용
=== 기술/도구 ===
내용
=== 자격증 ===
내용
=== 추가로 물어볼 것 ===
내용

본문:
{resume_text}
"""
    )
    section_msg = section_prompt.format(resume_text=resume_text)
    section_resp = llm.invoke(section_msg)
    resume_sections = section_resp.content.strip()

    # (3) 키워드 추출 (쉼표 구분)
    keyword_prompt = ChatPromptTemplate.from_template(
        """너는 위 이력서 요약문을 바탕으로 면접 질문을 만들 핵심 키워드만 추출한다.
아래 요약문을 보고 핵심 단어 5~10개만 뽑아라.
키워드만 쉼표(,)로 구분해서 출력해라.

요약문:
{summary}
"""
    )
    keyword_msg = keyword_prompt.format(summary=resume_summary)
    keyword_resp = llm.invoke(keyword_msg)

    parser = CommaSeparatedListOutputParser()
    resume_keywords = parser.parse(keyword_resp.content)

    # 최종 state 업데이트
    return {
        **state,
        "resume_summary": resume_summary,
        "resume_sections": resume_sections,
        "resume_keywords": resume_keywords,
    }
