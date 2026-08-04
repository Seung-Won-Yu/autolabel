# Landing page review

## Scope

- Source: `docs/index.html`, `docs/og.png`
- Interface: single-page developer-tool marketing site
- Primary user flow: value understanding → evidence → install → GitHub
- Review basis: 15 usability principles, desktop/light/dark/mobile browser rendering

## Findings resolved

| Severity | Finding | Resolution |
|---|---|---|
| 3 | The install CTA landed on commands that assumed the repository was already cloned, so a new visitor could not complete the primary task. | Added `git clone … && cd autolabel` as step 1 and kept setup/run as the next two steps (`docs/index.html:954`). |
| 2 | “First label in five minutes” was not backed by a measured setup time and could weaken trust when the 1.2GB model download takes longer. | Replaced it with outcome-based, verifiable install and first-batch copy (`docs/index.html:574`, `docs/index.html:1022`). |
| 2 | Static information cards lifted on hover, visually implying that they were clickable. | Removed interactive hover motion from non-interactive cards (`docs/index.html:309`). |
| 2 | Reveal content could remain hidden if the later interaction script failed after the early `.js` class was applied. | Added an independent three-second reveal watchdog; content also remains visible when JavaScript is disabled (`docs/index.html:528`). |
| 2 | Mobile navigation and Korean line breaking were fragile at the narrowest supported width. | Reduced mobile navigation to brand + a 44px GitHub target and enforced word-level Korean wrapping (`docs/index.html:92`, `docs/index.html:121`). |
| 1 | Dense-table and comparison controls exposed limited keyboard context. | Made the table a named focusable region and added live slider value text (`docs/index.html:907`, `docs/index.html:1053`). |

Outstanding severity 2–4 findings: **0**.

## Principle coverage

| Principle | Result |
|---|---|
| System status | Scroll progress, timeline progress, and reveal states are visible; no asynchronous submission exists on this static page. |
| Real-world match | Copy now describes the user task—folder in, model draft, exceptions reviewed—before implementation details. |
| Control and freedom | Internal navigation, skip link, native FAQ disclosure, and slider Home/End/arrow controls provide escape and direct movement. |
| Consistency | One primary/secondary CTA hierarchy, shared tokens, and role colors are used throughout. |
| Error prevention | Honest local/optional-cloud language and license FAQ prevent capability and cost assumptions. |
| Recognition | Visible workflow labels, install steps, examples, and evidence reduce recall requirements. |
| Efficiency | Sticky navigation, anchors, keyboard slider, and copyable three-step install support fast paths. |
| Minimalism | Decorative glow was replaced by a low-contrast coordinate grid; factual metrics no longer animate through false intermediate values. |
| Error recovery | Reveal watchdog and no-JS baseline keep the full message available if enhancements fail. |
| Help/documentation | Install requirements, model fallback, export options, FAQ, GitHub, and README links are adjacent to the relevant decision. |
| Affordances | Links and buttons have distinct hierarchy, focus states, 44px navigation CTA, and non-clickable cards no longer lift. |
| Structure | Header/main/footer landmarks, section headings, responsive grids, and focusable overflow table are present. |
| Accessibility | Korean language, semantic landmarks, skip link, visible focus, reduced motion, keyboard slider, role labels, and contrast-adjusted light tokens are present. |
| Perceptibility | Primary CTA, measurement labels, pass/fail/unsure text plus color, and section hierarchy are visually distinct. |
| Forgiveness | The static page remains readable without JavaScript and on 320px width without horizontal overflow. |

## Verification evidence

- Playwright: 1440px dark/light, 390px dark, 320px dark, reduced motion, JavaScript disabled
- All 8 internal anchors valid; 49/49 reveal targets visible
- Horizontal overflow: none in every tested viewport
- Comparison slider: arrow-key value change and accessible value text verified
- Skip link: first Tab target, visible at `y=8`
- Console/page errors: none
- OG image: 1200×630, 131KB
