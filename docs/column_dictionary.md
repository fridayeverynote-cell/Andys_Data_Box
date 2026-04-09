# 컬럼 역할 표

본 표는 프로젝트의 최종 사용 파일 기준 컬럼 역할표이다.

대상 파일:
- `continuous_dialogue_dialogue.csv`
- `continuous_dialogue_utterance.csv`
- `rag_documents.csv`
- `response_pairs.csv`

| 파일명 | 컬럼명 | 의미 | 활용 |
|---|---|---|---|
| continuous_dialogue_dialogue | dialogue_group_id | 대화 그룹 고유 ID | 대화 추적 |
| continuous_dialogue_dialogue | full_dialogue | 대화 전체 텍스트 | 대화 흐름 분석 |
| continuous_dialogue_dialogue | emotion_sequence | 감정 흐름 시퀀스 | 감정 변화 분석 |
| continuous_dialogue_dialogue | turn_count | 총 발화 수 | 대화 길이 분석 |

| continuous_dialogue_utterance | dialogue_group_id | 대화 그룹 고유 ID | 대화 연결 |
| continuous_dialogue_utterance | turn_index | 발화 순서 | 발화 흐름 파악 |
| continuous_dialogue_utterance | utterance | 실제 발화 텍스트 | 감정 분석 입력 |
| continuous_dialogue_utterance | emotion | 발화 감정 라벨 | 감정 분류 기준 |

| rag_documents | dialogue_id | 대화 고유 ID | 검색 결과 추적 |
| rag_documents | file_name | 원본 파일 식별 정보 | 원천 추적 |
| rag_documents | relation | 관계 유형 | 연인 데이터 필터 확인 |
| rag_documents | situation | 상황 요약 | 유사 사례 검색 핵심 |
| rag_documents | speaker_emotion | 화자 감정 | 감정 조건 검색 |
| rag_documents | listener_behavior | 청자 반응 요약 | 응답 방식 참고 |
| rag_documents | avg_rating | 평균 평점 | 품질 참고 |
| rag_documents | grade | 응답 등급 | 품질 구분 |
| rag_documents | speaker_texts | 화자 발화 모음 | 발화 패턴 참고 |
| rag_documents | listener_texts | 청자 발화 모음 | 응답 패턴 참고 |
| rag_documents | full_dialogue | 전체 대화 | 검색 근거 문서 |
| rag_documents | listener_empathy_tags | 공감 태그 | 공감 방식 참고 |
| rag_documents | final_speaker_change_emotion | 최종 감정 변화 | 대화 효과 참고, 기존 빈값은 unknown으로 처리 |
| rag_documents | risk_level | 갈등 위험도 | 위험도 기반 필터링 |
| rag_documents | turn_count | 발화 수 | 대화 길이 참고 |
| rag_documents | terminated | 대화 종료 여부 | 흐름 종료 판단 |

| response_pairs | dialogue_id | 대화 고유 ID | 원본 추적 |
| response_pairs | relation | 관계 유형 | 연인 데이터 필터 확인 |
| response_pairs | situation | 상황 요약 | 문맥 이해 |
| response_pairs | speaker_emotion | 화자 감정 | 감정 기반 추천 |
| response_pairs | context_before_response | 응답 직전 문맥 | 추천 답변 입력 맥락 |
| response_pairs | listener_response | 실제 청자 응답 | 추천 답변 예시 |
| response_pairs | listener_empathy | 응답 공감 유형 | 공감 스타일 분류, 기존 빈값은 미분류로 처리 |
| response_pairs | terminate | 응답 후 종료 여부 | 응답 결과 참고 |