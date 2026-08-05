# 구현 체크리스트

1. 기준 측정 ✅
   - 격리 DB로 `qa_e2e.py` 실행
   - 데스크톱·좁은 창·모바일 렌더와 콘솔 확인
2. Colab 학습 레인 ✅
   - 승인 데이터 전용 split zip endpoint
   - 노트북 파라미터 검증, test 평가, 사용자 단계 정리
   - zip·노트북 API 회귀 테스트
3. 통계 검수 ✅
   - 표본 전용 목록
   - 이미지별 정상/오류 기록, 전량 판정 전 제출 잠금
   - Playwright 사용자 흐름 테스트
4. 프런트 기반 ✅
   - 한국어 문서 메타, 포커스·live region·dialog
   - 좁은 화면 레이아웃과 핵심 컨트롤 접근성
5. 검증 ✅
   - 전체 pytest, unit, lint, build, Playwright
   - 실제 종단 QA 재실행과 UX 사후 감사
