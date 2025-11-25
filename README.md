# MUSE AI - Enterprise Chatbot Platform

<p align="center">
  <img src="assets/muse-logo.png" alt="MUSE AI Logo" width="200"/>
</p>

MUSE는 프로덕션 레디 엔터프라이즈 AI 챗봇 플랫폼입니다. 멀티 LLM 지원, RAG 파이프라인, 실시간 스트리밍, 사용자 인증, 관리자 대시보드를 포함한 완전한 솔루션입니다.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

## Features

### Core
- **Multi-LLM Support** - OpenAI GPT-4, Anthropic Claude 동시 지원
- **RAG Pipeline** - 문서 기반 질의응답 (PDF, DOCX, TXT)
- **Streaming Response** - Server-Sent Events 기반 실시간 스트리밍
- **Conversation Memory** - Redis 기반 세션 및 대화 히스토리 관리
- **Function Calling** - 외부 API 연동 및 도구 사용

### Enterprise
- **Authentication** - JWT 기반 인증 + API Key
- **Rate Limiting** - 사용자/API별 요청 제한
- **Multi-tenancy** - 조직별 독립 환경 지원
- **Audit Logging** - 모든 대화 및 액션 로깅

### Scalability
- **Async Architecture** - 비동기 처리로 높은 동시성 지원
- **Queue System** - Celery + Redis 기반 백그라운드 작업
- **Caching** - 응답 캐싱으로 비용 절감
- **Docker Ready** - Docker Compose로 원클릭 배포

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Load Balancer                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                 MUSE API Gateway                             │
│              (FastAPI + Authentication)                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│  Chat Service │ │  RAG Service  │ │ Admin Service │
│   (Streaming) │ │  (Embedding)  │ │  (Dashboard)  │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Message Queue                             │
│                  (Redis + Celery)                           │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
│  PostgreSQL   │ │     Redis     │ │   ChromaDB    │
│   (Primary)   │ │   (Cache)     │ │  (Vectors)    │
└───────────────┘ └───────────────┘ └───────────────┘
```

## Quick Start

### Docker (Recommended)

```bash
# 환경변수 설정
cp .env.example .env
# .env 파일에서 API 키 설정

# 컨테이너 실행
cd docker && docker-compose up -d

# 접속
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Local Development

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 데이터베이스 마이그레이션
alembic upgrade head

# 개발 서버 실행
uvicorn app.main:app --reload --port 8000
```

## Project Structure

```
muse-chatbot/
├── app/
│   ├── api/v1/endpoints/   # API 엔드포인트
│   ├── core/               # 설정, 보안, 예외
│   ├── models/             # SQLAlchemy 모델
│   ├── schemas/            # Pydantic 스키마
│   ├── services/           # 비즈니스 로직
│   │   ├── llm/            # LLM 프로바이더
│   │   └── rag/            # RAG 파이프라인
│   ├── db/                 # 데이터베이스
│   └── main.py             # FastAPI 앱
├── docker/                 # Docker 설정
├── tests/                  # 테스트
└── requirements.txt
```

## API Reference

### Authentication

```bash
# 회원가입
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123"}'

# 로그인
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "securepass123"}'
```

### Chat with MUSE AI

```bash
# 일반 채팅
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "안녕하세요, MUSE!",
    "model": "gpt-4-turbo-preview"
  }'

# 스트리밍 채팅
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{"message": "긴 글을 작성해줘"}'
```

### RAG (Document Q&A)

```bash
# 문서 업로드
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer {token}" \
  -F "file=@document.pdf"

# 문서 기반 질의
curl -X POST http://localhost:8000/api/v1/chat/rag \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "이 문서에서 핵심 내용이 뭐야?",
    "document_ids": ["doc_123"]
  }'
```

## Configuration

주요 환경변수는 `.env.example` 참고

```env
# LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/muse

# Redis
REDIS_URL=redis://localhost:6379/0
```

## License

MIT License
