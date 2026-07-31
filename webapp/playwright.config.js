import { defineConfig } from '@playwright/test'

// 프론트 회귀 잠금. 실사용 테스트에서 손으로 찾은 결함(포인터 상호작용, 배경
// 클릭 확정, 임포트 후 빈 캔버스)이 다시 살아나지 않게 한다.
// 모델 추론이 필요한 흐름은 여기 넣지 않는다 — scripts/qa_*_e2e.py 담당.
const ROOT = new URL('..', import.meta.url).pathname
const E2E_DIR = `${ROOT}data/e2e`

// DB를 지워서 격리하지 않는다. 이 설정 파일은 메인 프로세스와 워커에서 각각
// 평가되므로, 여기서 지우면 이미 켜진 백엔드의 DB를 걷어찬다 (globalSetup도
// webServer 다음에 돌아 마찬가지). 대신 테스트가 매번 고유한 이름을 쓴다.

export default defineConfig({
  testDir: './e2e',
  timeout: 30_000,
  expect: { timeout: 7_000 },
  fullyParallel: false,   // 백엔드 DB 하나를 공유한다
  workers: 1,
  reporter: [['list']],
  use: {
    // vite는 localhost(::1)에만 바인딩한다 — 127.0.0.1로는 붙지 않는다
    baseURL: 'http://localhost:5273',
    viewport: { width: 1440, height: 900 },
    trace: 'retain-on-failure',
  },
  webServer: [
    {
      // 실제 DB를 건드리지 않게 격리된 DB·업로드 경로로 띄운다.
      // AUTOLABEL_NO_MODELS=1은 필수다 — 프론트 e2e는 SAM·GDINO가 필요 없는데
      // 이미지를 열 때마다 임베딩을 계산하려 들어, 개발용 서버와 동시에
      // SAM ViT-L(1.2GB)을 MPS에 올리다 프로세스가 통째로 죽었다.
      command: `AUTOLABEL_DB=${E2E_DIR}/e2e.db AUTOLABEL_DATA=${E2E_DIR}/uploads `
        + 'AUTOLABEL_NO_MODELS=1 '
        + `${ROOT}.venv/bin/python -m uvicorn server.main:app --port 8991`,
      cwd: ROOT,
      port: 8991,
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: 'npm run dev -- --port 5273 --strictPort',
      url: 'http://localhost:5273',
      reuseExistingServer: false,
      timeout: 60_000,
      env: { VITE_API_PORT: '8991' },
    },
  ],
})
