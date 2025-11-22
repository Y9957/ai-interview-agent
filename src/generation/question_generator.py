# src/generation/question_generator.py

from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# ============================================================
# generate_question 
# ============================================================

def generate_question(state: Dict[str, Any]) -> Dict[str, Any]:

    # ---------- 1) 상태 읽기 ----------
    summary      = state.get("resume_summary", "")
    keywords     = ", ".join(state.get("resume_keywords", []))
    q_strategy   = state.get("question_strategy", {}) or {}
    prev_q       = (state.get("current_question") or "").strip()
    prev_a       = (state.get("current_answer") or "").strip()
    eval_list    = state.get("evaluation", []) or []
    focus_area   = state.get("current_strategy") or (
        "경력 및 경험" if "경력 및 경험" in q_strategy else (next(iter(q_strategy.keys()), "경력 및 경험"))
    )

    # 최근 평가 요약(우리 스키마: "질문과의 연관성", "답변의 구체성")
    if eval_list and isinstance(eval_list[-1], dict):
        last = eval_list[-1]
        eval_brief = f"질문과의 연관성: {last.get('질문과의 연관성','')}, 답변의 구체성: {last.get('답변의 구체성','')}"
    else:
        eval_brief = "이전 답변에 대한 평가는 제공되지 않았습니다."

    # ---------- 2) 코퍼스 구성 ----------
    corpus_texts, metadatas = [], []

    # 전략 예시질문(키 이름: '예시질문')
    for area, cfg in q_strategy.items():
        for q in (cfg.get("예시질문", []) or []):
            if q:
                corpus_texts.append(q)
                metadatas.append({"source": "strategy", "area": area})

    # 히스토리 질문
    for turn in (state.get("conversation", []) or []):
        q = turn.get("question", "")
        if q:
            corpus_texts.append(q)
            metadatas.append({"source": "history", "area": "history"})

    # ---------- 3) 유사 질문 검색 ----------
    similar_refs = []
    if corpus_texts:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        except TypeError:
            embeddings = OpenAIEmbeddings()
        vs = Chroma.from_texts(texts=corpus_texts, embedding=embeddings, metadatas=metadatas)

        query_text = prev_q or (keywords or summary[:200])
        k = min(3, len(corpus_texts))
        docs = vs.similarity_search(query_text, k=k)
        similar_refs = [d.page_content for d in docs]

    refs_block = "\n".join(f"- {r}" for r in similar_refs) if similar_refs else "- (참고 질문 없음)"

    # ---------- 4) LLM 프롬프트 ----------
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.5)
    prompt = ChatPromptTemplate.from_template(
        """
        당신은 전문 면접관입니다. 아래 정보를 바탕으로 지원자의 사고력/문제해결/기술적 깊이를 더 확인할 수 있도록
        간결하고 명확한 '심화 질문 1개'만 한국어로 작성하세요. (출력은 질문 한 문장만)

        [면접 포커스 영역]
        {focus_area}

        [이력서 요약]
        {summary}

        [키워드]
        {keywords}

        [이전 질문]
        {prev_q}

        [이전 답변]
        {prev_a}

        [이전 답변 평가 요약]
        {eval_brief}

        [참고용 유사 질문(수정 금지, 생성에만 참고)]
        {refs_block}

        요구사항:
        - 이전 답변의 부족한 부분(연관성/구체성)을 보완하도록 유도
        - 정량 근거(지표/수치/기간 등)나 구체 사례를 끌어내도록 구성
        - '어떻게/무엇을 근거로/어떤 기준으로' 형태의 꼬리질문 권장
        - 반드시 한 문장·질문부호로 끝낼 것
        """
    )

    resp = (prompt | llm).invoke({
        "focus_area": focus_area,
        "summary": summary,
        "keywords": keywords,
        "prev_q": prev_q,
        "prev_a": prev_a,
        "eval_brief": eval_brief,
        "refs_block": refs_block
    })
    new_q = (resp.content or "").strip()

    # ---------- 5) 간단 품질 체크 & 폴백 ----------
    used_questions = list(state.get("used_questions", []))
    candidates     = list((q_strategy.get(focus_area, {}) or {}).get("예시질문", []))

    if (not new_q.endswith("?")) or (len(new_q) < 8) or (new_q in used_questions):
        fallback_pool = [q for q in candidates if q not in used_questions] or candidates
        new_q = (fallback_pool or [
            "이 경험이 현재 지원 직무와 어떻게 연결되는지, 정량 지표와 함께 한 문장으로 설명해 주실 수 있나요?"
        ])[0]

    # ---------- 6) 커버리지/사용질문 갱신 ----------
    coverage = dict(state.get("strategy_coverage", {}))
    coverage[focus_area] = coverage.get(focus_area, 0) + 1

    if new_q not in used_questions:
        used_questions = used_questions + [new_q]

    return {
        "current_question": new_q,
        "current_answer": "",
        "strategy_coverage": coverage,
        "used_questions": used_questions,
    }


# ============================================================
# summarize_interview 
# ============================================================

def summarize_interview(state: Dict[str, Any]) -> Dict[str, Any]:
    conversations     = state.get("conversation", []) or []
    evaluations       = state.get("evaluation", []) or []
    question_strategy = state.get("question_strategy", {}) or {}

    allowed_sections = list(question_strategy.keys()) or [
        "경력 및 경험", "동기 및 커뮤니케이션", "논리적 사고", "기술 역량 및 전문성", "성장 가능성 및 자기주도성"
    ]

    by_section = {sec: [] for sec in allowed_sections}
    for i, turn in enumerate(conversations):
        sec = (turn.get("strategy") or (allowed_sections[0] if allowed_sections else "")).strip()
        if sec not in by_section:
            sec = allowed_sections[0] if allowed_sections else sec

        q = turn.get("question", "")
        a = turn.get("answer", "")
        ev = evaluations[i] if i < len(evaluations) and isinstance(evaluations[i], dict) else {}
        by_section[sec].append({"q": q, "a": a, "ev": ev})

    section_materials = []
    for sec in allowed_sections:
        items = by_section.get(sec, [])
        if not items:
            section_materials.append(f"[{sec}] (해당 턴 없음)")
            continue
        lines = [f"[{sec}]"]
        for j, it in enumerate(items, 1):
            ev = {k: v for k, v in (it["ev"] or {}).items() if k != "question_index"}
            ev_text = ", ".join(f"{k} : {v}" for k, v in ev.items()) if ev else "평가 없음"
            lines.append(f"- ({j}) Q: {it['q']}")
            lines.append(f"      A: {it['a']}")
            lines.append(f"      평가: {ev_text}")
        section_materials.append("\n".join(lines))
    materials_block = "\n\n".join(section_materials) if section_materials else "(대화 없음)"

    sections_block = "\n\n".join(
        f"[{sec}]\n- 답변 요약:\n- 강점:\n- 약점:\n- 평가 경향: 질문과의 연관성 : <상/중/하>, 답변의 구체성 : <상/중/하>"
        for sec in allowed_sections
    )

    prompt = f"""
당신은 아래 '섹션별 자료'를 바탕으로 **전략별 피드백 보고서**를 작성합니다.

[허용 섹션(이 목록 외 섹션 절대 금지)]
- {", ".join(allowed_sections)}

[섹션별 자료]
{materials_block}

[작성 규칙]
- 반드시 아래 '출력 형식'을 그대로 사용(제목/구분선/섹션명 고정, 추가 섹션 금지).
- 각 섹션에는 해당 섹션으로 분류된 Q/A만 반영. 해당 턴이 없으면 '답변 요약/강점/약점'에 '해당 없음'으로 기입.
- '강점'은 실제 기록에서 확인되는 근거(수치/기간/지표/구체적 사례)가 없는 경우 '해당 없음'으로 작성.
- 최근 평가가 대부분 '하'인 섹션은 강점을 쓰지 말고 '해당 없음'으로.
- 각 섹션 마지막 줄에 '평가 경향'을 다음 형식으로 반드시 기입:
  - 예) 평가 경향: 질문과의 연관성 : 중, 답변의 구체성 : 하

[출력 형식]
=======================================
[전략별 피드백]

{sections_block}

=======================================
[종합 피드백]
- 전체 인상:
- 핵심 강점:
- 핵심 보완점:
"""

    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.3)
    summary_text = llm.invoke(prompt).content.strip()

    print("\n" + "=" * 60)
    print("[면접 피드백 보고서 요약 결과]")
    print("=" * 60)
    print(summary_text)
    print("=" * 60 + "\n")

    return {
        **state,
        "summary_report": summary_text,
        "next_step": "end",
    }


# ============================================================
# route_after_reflect / route_after_decide 
# ============================================================

def route_after_reflect(state: Dict[str, Any]) -> str:
    return "re_evaluate" if state.get("next_step") == "re_evaluate" else "decide"


def route_after_decide(state: Dict[str, Any]) -> str:
    return "summarize" if state.get("next_step") == "end" else "generate"

