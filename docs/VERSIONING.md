# Versioning Rules (youngjaedev fork)

이 문서는 upstream (oh-my-claudecode)과 구분되는 fork 버전 관리 규칙을 정의합니다.

## 버전 형식

```
MAJOR.MINOR.PATCH-youngjaedev.BUILD
```

**예시:** `3.5.7-youngjaedev.1`

| 부분 | 설명 |
|------|------|
| `MAJOR.MINOR.PATCH` | upstream 버전 기준 |
| `-youngjaedev` | fork 식별자 |
| `.BUILD` | fork 내 빌드 번호 |

## 버전 업데이트 규칙

### Fork 내 변경 시

| 상황 | 액션 | 예시 |
|------|------|------|
| 새 기능 추가 | BUILD +1 | `3.5.7-youngjaedev.1` → `3.5.7-youngjaedev.2` |
| 버그 수정 | BUILD +1 | `3.5.7-youngjaedev.2` → `3.5.7-youngjaedev.3` |
| 문서 변경만 | BUILD 유지 또는 +1 | 선택적 |

### Upstream 머지 시

| 상황 | 액션 | 예시 |
|------|------|------|
| upstream `3.5.7` 머지 | PATCH+1, BUILD 리셋 | `3.5.7-youngjaedev.3` → `3.5.8-youngjaedev.1` |
| upstream `3.6.0` 머지 | MINOR+1, BUILD 리셋 | `3.5.x-youngjaedev.x` → `3.6.0-youngjaedev.1` |
| 충돌 해결 포함 | 동일 | - |

## SemVer 호환성

- `-youngjaedev.N` 접미사는 SemVer pre-release로 취급됨
- 비교 순서: `3.5.7` > `3.5.7-youngjaedev.1` (pre-release가 낮음)
- 하지만 fork는 upstream과 별도 배포이므로 문제 없음

## 파일 위치

버전은 다음 파일에서 관리:

```
.claude-plugin/plugin.json → "version" 필드
```

## 히스토리

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| `3.5.7-youngjaedev.1` | 2026-01-27 | humanizer skill/agent 추가 |
| `3.5.6` | - | upstream 기준 시작점 |
