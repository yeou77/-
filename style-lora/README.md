# style-lora

특정 작가(예: 건어물녀 — 《꽃은 미끼야》, 《메리 사이코》)의 문체·리듬·어휘를 흉내내는
초안 생성기를 QLoRA로 만드는 파이프라인. NovelAI의 커스텀 모듈 학습이 신형 모델에서
막힌 뒤로 남는 현실적인 방법은, 클라우드 GPU에서 얇은 LoRA 어댑터를 학습해서
파일로 소유하는 것 — 서비스가 사라져도 다시 돌릴 수 있는 자산으로 남는다.

**이 파이프라인이 하는 일**: 문장 호흡·어휘 선택·대사 톤을 넘겨받는 초안 생성기를
만든다. 그 작가의 플롯 설계나 캐릭터 사고방식까지 이식하진 않는다 — 초안을 뽑고
직접 살을 붙이는 용도다.

## 중요: 저작권 있는 원문은 이 저장소에 커밋하지 않는다

`data/raw/`, `data/processed/`, `outputs/` 는 `.gitignore`에 이미 등록되어 있다.
학습에 쓸 txt는 로컬/클라우드 디스크에만 두고, 절대 git에 커밋하거나 채팅으로
붙여넣지 않는다. 학습된 LoRA 가중치도 개인 용도로만 보관 — 특정 생존 작가의
문체를 학습해 상업적으로 배포하는 것은 법적 회색지대다. 초안 생성에만 쓰고,
발표하는 최종 문장은 네 것으로 바꾸는 습관을 들일 것.

## 준비물

- 정제된 본문 txt 2개 (작가의 말/후기/공지 제거된 순수 본문) → `data/raw/`에 각각
  `book1.txt`, `book2.txt` 식으로 넣기 (파일명은 자유)
- 클라우드 GPU 계정: Google Colab(가장 쉬움, T4/A100), RunPod, Vast.ai 중 하나
  - 8B급 모델 QLoRA면 16~24GB VRAM GPU로 충분 (Colab Pro의 A100, 또는 RunPod
    RTX 3090/4090 인스턴스)
  - 두 권 분량이면 학습 자체는 보통 1~3시간, 비용은 GPU 대여료 기준 몇 천 원대

## 사용 순서 (가장 쉬운 경로)

두 소설을 통째로 `data/raw/`에 넣고, Colab에서 노트북을 위에서부터 실행하면 끝난다.
챕터/문단을 직접 나누거나 평가용 문단을 손으로 고르는 과정은 없음 — 전부 자동.

```bash
cd style-lora
pip install -r requirements.txt

# data/raw/에 book1.txt, book2.txt (소설 통째로) 넣은 뒤:
python scripts/preprocess.py auto
```

- `data/raw/*.txt`를 챕터/문단 단위로 자동 분리하고, 평가용 문단 20개를 무작위로
  떼어(`--eval-n`으로 개수 조절) 나머지로 `data/processed/train_corpus.txt`(학습용)와
  `data/processed/eval_set.jsonl`(평가용)을 바로 만든다. 다음 단계인 학습으로 바로 넘어가면 됨.

### (선택) 더 정교하게: 평가 문단을 직접 고르고 싶다면

무작위 대신 "이 호흡이다" 싶은 문단을 직접 골라 평가 기준으로 삼고 싶으면 이 방식을 대신 쓴다:

```bash
python scripts/preprocess.py candidates   # 책 x 지문/대사/심리별 후보 추출
# data/processed/eval_candidates.jsonl을 열어 ~30개 id를 골라 eval_selected.txt에 한 줄씩 저장
python scripts/preprocess.py build --eval-ids eval_selected.txt
```

두 방식 모두 결과물은 동일하게 `train_corpus.txt` / `eval_set.jsonl`이라 이후 단계는 같다.

### 클라우드에서 QLoRA 학습

로컬 노트북 내장 그래픽으로는 불가능 — Colab/RunPod에서 실행. `notebook/colab_quickstart.ipynb`
참고하거나 직접:

```bash
python scripts/train_lora.py \
    --model_name beomi/Llama-3-Open-Ko-8B-Instruct-preview \
    --train_file data/processed/train_corpus.txt \
    --output_dir outputs/style-lora
```

- 두 작품을 하나의 코퍼스로 합쳐서 학습 (같은 작가라 문체 결이 같음).
- 다른 작가 텍스트는 절대 섞지 않기 — 섞으면 '평균 문체'가 되어 개성이 죽는다.
- `--model_name`은 한국어 창작에 버티는 다른 오픈 모델로 바꿔도 된다 (예:
  `yanolja/EEVE-Korean-10.8B-v1.0`, `MLP-KTLim/llama-3-Korean-Bllossom-8B`).
  베이스 모델을 자주 갈아타지 말고, 갈아탈 땐 같은 데이터로 다시 학습할 것.
  사용 전 해당 모델의 라이선스(상업적 이용 가능 여부 등)를 반드시 확인.
- 기본값은 4bit QLoRA, rank 16 — VRAM 24GB 이하에서도 돌아가도록 설정.

### (대안) Together.ai로 학습 — GPU 클릭 없이

직접 GPU를 붙잡고 있기 귀찮으면 `notebook/colab_together.ipynb`를 대신 쓴다.
GPU 런타임도 필요 없다 (Colab은 그냥 API 요청을 보내는 통로 역할만 함, 실제 학습은
Together 클라우드에서 돎). 셀 하나(`scripts/train_together.py`)가 업로드 → 학습 시작
→ 진행 확인 → 결과 다운로드까지 전부 처리한다.

- Together 계정에 결제수단/크레딧 등록 필요, API Keys 메뉴에서 키 발급.
- 사용 전 Together의 fine-tuning 지원 모델 목록에 `--model`로 쓸 모델이 있는지 확인.
- 학습마다 비용이 청구된다 (RUN_LOG.md에 기록해두면 좋음).
- 결과 어댑터를 다운로드하지 못하는 경우도 있음 — 그때는 RUN_LOG.md의 이식성
  체크리스트를 보고, 같은 코퍼스로 로컬/Colab GPU 학습(`train_lora.py`)으로 넘어간다.

### 평가: base vs +lora 비교

```bash
python scripts/eval.py --lora_dir outputs/style-lora --out outputs/eval_report.md
```

떼어둔 평가 문단(자동이든 직접 골랐든)의 첫 문장을 프롬프트로 써서, 베이스 모델과
LoRA 적용 모델의 이어쓰기를 나란히 리포트로 뽑는다. 점수는 자동으로 매기지 않는다 —
읽어보고 "이 쪽 문체네" 싶은지 직접 판단.

## 기대치

| 방식 | 체감 |
|---|---|
| 프롬프트에 샘플만 넣은 범용 챗봇 | 문장 몇 개는 비슷하지만 곧 모델 특유의 톤으로 복귀 |
| 이 파이프라인의 작은 LoRA | 문장 호흡·어휘·대사 톤이 초안 단계부터 드러남 |
| 최종 결과물 | 여전히 사람이 손봐야 완성 — 플롯/캐릭터 사고는 이식되지 않음 |

## 장기 운용 원칙

- **작가 하나 = 어댑터 하나.** 다른 작가를 추가하려면 새 `data/raw/`로 별도 학습.
- 학습 레시피(전처리 스크립트, 학습 스크립트, 하이퍼파라미터)와 `outputs/`의 어댑터
  파일을 백업해두면, 베이스 모델이 바뀌거나 서비스가 사라져도 다시 만들 수 있다 —
  이게 구독형 서비스(NovelAI 모듈 등)와 다른 점.
- 재학습은 자주 하는 게 아니라 데이터가 늘 때(신작 추가 등) 가끔 클라우드 GPU를
  몇 시간 빌리는 구조 — 월 구독료보다 통제하기 쉽다.
