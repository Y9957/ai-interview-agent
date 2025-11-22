# 🤖 AI Interview Agent

이 프로젝트는 OpenAI 기반 LLM과 LangGraph를 활용하여  
**이력서 분석 → 질문 전략 생성 → 질문/답변 평가 → Reflection → 심화 질문 → 최종 피드백 리포트**까지  
면접 전 과정을 자동화한 **AI 면접관 Agent(v2.0)** 입니다.

---

## 🚀 주요 기능

### ✔ 1. 이력서 분석 & 요약
- PDF/DOCX 이력서를 입력받아 텍스트 추출
- 핵심 경력/기술 요약
- 키워드 및 섹션 분류 생성

### ✔ 2. 질문 전략 생성
- 생성된 요약/키워드를 기반으로 5개 전략 자동 생성  
  (경력/경험, 동기/커뮤니케이션, 논리적 사고, 기술 역량, 성장 가능성)

### ✔ 3. 질문 & 답변 평가
- 답변의 **연관성/구체성**을 LLM으로 평가  
- 평가 결과는 ‘상/중/하’로 기록

### ✔ 4. Reflection(재평가)
- 답변 길이 부족, 구체성 부족, 평가 모순 시 자동 재평가 수행  
- re-evaluate 후 다음 단계 진행

### ✔ 5. 심화 질문 생성
- 벡터 기반 유사 질문 탐색  
- 부족한 부분을 보완하도록 질문 설계  
- fallback 예시질문 로직 포함

### ✔ 6. 전략별 최종 평가 리포트
- 전략별 섹션 요약  
- 강점 / 약점 / 평가 경향  
- 전체 종합 피드백 생성

### ✔ 7. Gradio UI 지원
- 이력서 업로드 → 면접 진행 → 보고서 출력  
- 전체 과정을 웹에서 실행 가능

---

## 🧩 폴더 구조

```bash
ai-interview-agent/
└── src/
     ├── resume/
     │     └── resume_parser.py          # analyze_resume
     │
     ├── strategy/
     │     └── strategy_generator.py      # generate_question_strategy
     │
     ├── evaluation/
     │     └── evaluator.py               # evaluate / reflect / re_evaluate
     │
     ├── generation/
     │     └── question_generator.py      # generate_question + summarize_interview + route functions
     │
     ├── decision/
     │     └── decider.py                 # decide_next_step
     │
     └── graph/
           └── agent_v2.py                # preProcessing + LangGraph 전체 파이프라인

app.py                                      # Gradio UI
run.py                                      # CLI 테스트용
requirements.txt
README.md
```

## ⚙️ 실행 방법 

### 1) 환경 변수 설정
```bash
export OPENAI_API_KEY="api_key"
```

Windows PowerShell:
```powershell
setx OPENAI_API_KEY "api_key"
```

### 2) 패키지 설치
```bash
pip install -r requirements.txt
```

### 3) CLI(터미널) 모드 실행
```bash
python run.py
```

### 4) Gradio Web UI 실행
```bash
python app.py
```

---

## 🧠 버전 설명

v2.0 (현재 버전 / 최종본)

질문 전략 딕셔너리 구조

Evaluation + Reflection + Re-evaluate 체계

LangGraph 기반 파이프라인

심화 질문 생성 강화

Gradio UI 추가

전략별 섹션 기반 최종 보고서 자동 생성
