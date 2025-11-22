# src/decision/decider.py

from typing import Dict, Any

def decide_next_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    (3) 인터뷰 진행 검토 : 고도화
      - 전체 Q&A 5회 도달 → end  (우선순위 0: 명확한 종료)
      - 첫 라운드 보장: 아직 1회 미커버 전략이 있으면 그 전략으로 전환
      - (첫 라운드 이후) 최근 평가에 '하' 존재 → additional_question(현재 전략 유지)
      - (그 외) 다음 전략으로 전환
      - 참고: '모든 전략 1회 이상 커버 → end'는 5턴 이후에만 적용(조기 종료 방지)
    """
    # ----- 입력 상태 읽기 -----
    strategies = list(state.get("question_strategy", {}).keys())
    coverage   = state.get("strategy_coverage", {}) or {}
    conv_len   = len(state.get("conversation", []) or [])
    evals      = state.get("evaluation", []) or []
    cur_strat  = state.get("current_strategy", strategies[0] if strategies else "")

    # 전략이 없으면 종료
    if not strategies:
        return {"next_step": "end"}

    # (우선순위 0) Q&A 5회 제한 → 종료
    if conv_len >= 5:
        return {"next_step": "end"}

    # (우선순위 1) 첫 라운드 보장: 아직 1회도 안 한 전략이 있으면 그 전략으로 전환
    not_covered = [s for s in strategies if coverage.get(s, 0) < 1]
    if not_covered:
        # 현재 전략을 이미 1회 했으면 다음 미커버 전략으로 전환
        if coverage.get(cur_strat, 0) >= 1:
            next_strat = not_covered[0]
            return {
                "current_strategy": next_strat,
                "decision": "next_strategy",
                "next_step": "generate",
            }
        # 아직 현재 전략이 1회 미만이면 그대로 진행(= 이번 턴에 현재 전략으로 질문 생성)

    # (옵션) 모든 전략 커버 완료 → 종료 (단, 5턴 이후에만 적용해 조기 종료 방지)
    if all(coverage.get(s, 0) >= 1 for s in strategies) and conv_len >= 5:
        return {"next_step": "end"}

    # (첫 라운드 이후) 최근 평가 확인
    last_eval = evals[-1] if evals else {}
    rel = last_eval.get("질문과의 연관성", "중")
    spc = last_eval.get("답변의 구체성", "중")
    if not not_covered:  # 첫 라운드가 끝난 상태에서만 '하'에 따른 꼬리질문 허용
        if rel == "하" or spc == "하":
            return {
                "decision": "additional_question",
                "next_step": "generate",
            }

    # (일반) 다음 전략으로 전환
    try:
        idx = strategies.index(cur_strat)
    except ValueError:
        idx = -1
    next_strat = strategies[(idx + 1) % len(strategies)]
    return {
        "current_strategy": next_strat,
        "decision": "next_strategy",
        "next_step": "generate",
    }
