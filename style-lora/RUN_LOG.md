# 학습 시도 기록

학습을 어디서(Colab/RunPod/Together 등) 돌리든, 재현하려면 이 값들이 맞아야 한다.
새로 학습할 때마다 아래 표에 한 줄씩 추가. 결과물(어댑터 파일)을 다른 곳으로 옮기지
못하더라도, 이 기록 + `data/processed/train_corpus.txt`만 있으면 같은 설정으로
어디서든 재현(재학습)할 수 있다.

| 날짜 | 실행처 | base_model | lora_r | lora_alpha | epochs | 코퍼스 버전* | 다운로드 가능? | 결과 판단 |
|---|---|---|---|---|---|---|---|---|
| 2026-08-27 | Together.ai (job_id=ft-59fe1f00-367d) | Qwen/Qwen3.5-9B | 16 | 32 | 3 | 책 2권(꽃은 미끼야+메리싸이코) v1 | 확인 중 (SDK 다운로드 함수 이름 불일치, 대시보드에서 확인) | 평가 전 |

\* 코퍼스 버전: `data/raw/`에 넣은 책이 바뀌면(신작 추가 등) train_corpus.txt도 달라지므로,
간단히 "책 2권(꽃은 미끼야+메리싸이코) v1" 식으로 구분 기록.

## 이식성 체크리스트 (외부 서비스에서 학습할 때)

- [ ] base_model이 그 서비스의 자체 전용 모델이 아니라 **공개 HuggingFace 모델**인지
      (예: `beomi/Llama-3-Open-Ko-8B-Instruct-preview`) — 아니면 나중에 로컬에서
      같은 베이스를 못 구해서 재현 자체가 안 됨
- [ ] 어댑터를 실제로 다운로드할 수 있는지 (표준 PEFT 포맷: `adapter_config.json` +
      `adapter_model.safetensors`)
- [ ] `lora_r`, `lora_alpha`, `target_modules`를 기록해뒀는지 — 다운로드가 안 돼서
      로컬 재학습으로 넘어갈 때 이 값들을 그대로 맞춰야 같은 결과가 나옴

세 개 다 맞으면 그대로 가져다 쓰면 되고, 하나라도 안 맞으면 같은 코퍼스로
`scripts/train_lora.py`를 로컬/다른 GPU에서 돌려서 재현하면 된다 — 데이터 준비를
다시 할 필요는 없다.
