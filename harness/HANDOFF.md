# FairGate 재실험 — 인수인계

> 새 서버(RTX6000)에서 이어서 작업하기 위한 문서.
> 2026-09-08-09, RTX5090에서 작성.
> **새 Claude 세션에서 이 파일을 먼저 읽어주세요.**

---

## 1. 프로젝트 상황

FairGate = GNN 공정성 프레임워크. 제출 이력:

- **NeurIPS 2026 제출 → 낮은 점수.** 원인: φ(v)를 "node-level fairness risk score"라고 주장했는데 식별 불가능한 양이었음.
- **WSDM 2027 준비 중 중단.** 프레이밍은 크게 개선됨(risk 주장 철회, "allocation" 축으로 재정의, 통제군 추가). 그러나 실험 오류를 발견해 제출하지 않음.
- **현재 목표**: 실험을 처음부터 제대로 설계해서 재수행. ICLR 2027은 마감(2026-09-18 abstract / 09-25 paper)이 촉박해 비현실적. **ICML 2027(2027년 1월 말)을 타겟, ICLR 2028 백업.**

### 논문의 주장 (Phase 0에서 확정)

**버린 주장 (사실이 아님):** "기존 fair GNN 연구는 전부 node-agnostic penalty를 쓴다."

원문 확인 결과 반례가 여럿 있음:

| 연구 | 무엇을 비균일하게 하는가 | 세부 |
|---|---|---|
| **FairGB** (KDD'24) | (y,s) **4개 subgroup 단위** gradient 기반 re-weighting | subgroup 내 상수 → **위상 정보 0**. 데이터셋 3개(German/Bail/Credit)만 사용 |
| **BIND** (AAAI'23) | **노드별** influence 추정 후 **harmful node 삭제** (이산 0/1) | φ(v)의 하드 버전. 데이터셋 4개가 우리와 겹침 |
| **ComFairGNN** (PAKDD'25) | community별 local **class**-homophily 상하위 15개 coreset 선택 | sensitive homophily 아님 |
| FairDrop / FairAGG | edge 단위 (sensitive homophily / Shapley) | |

**확정 주장 (참, 검증됨):**

> 비균일 배분은 이미 존재한다. 그러나 **기존 방법은 전부 배분 기준을 사전에 하나로 고정한다.**
> 어떤 기준이 유효한지가 그래프의 sensitive-group topology에 따라 달라진다는 사실,
> 그리고 그것이 **학습 전에 판별 가능하다**는 사실은 다뤄진 바 없다.

기여 3층 (가중치를 다르게 둘 것):

| | 내용 | 새로움 |
|---|---|---|
| D1 | group→node, discrete→continuous bounded | **낮음.** 구현 세부로 강등 |
| **D2** | **배분 기준의 선택을 그래프 통계에 조건화** | **높음 — 이게 논문** |
| **D3** | **배분이 무의미한 조건을 학습 전에 판정** | **높음 — top-venue 요소** |

---

## 2. 발견 1 — Credit 데이터 사고 (해결됨)

`data/credit/credit_edges.txt`가 **논문의 그래프가 아니었음.**

| | E | deg | h | r_b | δ_deg | regime |
|---|---|---|---|---|---|---|
| 발견 당시 | 304,754 | 10.16 | 0.891 | 0.442 | 0.106 | clustered ✗ |
| **논문 Table 5/6** | **2,873,716** | **95.79** | **0.960** | **0.677** | **0.315** | **degree-skewed** |

- Credit은 **유일한 degree-skewed 세팅** → 4개 regime 중 하나가 통째로 사라졌던 상태
- Credit은 FairGate가 최고 성적(ΔDP=ΔEO=0.009)을 낸 곳이자 ablation 근거
- **해결**: `FairGB-main/dataset/credit.zip`의 원본이 논문 통계 5개를 정확히 재현 → 교체 완료
- 잘못된 파일은 `data/credit/_quarantine_wrong_graph/`에 보존 (`README.txt`에 경위 기록)
- **현재 9개 세팅 전부 논문 Table 5/6과 완전 일치** (`harness/e0_diagnostics/provenance.csv`)
- `core/datasets.py`가 이제 불일치 시 예외를 던짐 → 재발 불가

---

## 3. 발견 2 — 논문의 핵심 메커니즘이 틀림

원고의 인과 사슬:
```
r_b → 1  ⟹  w_bdry가 상수  ⟹  순위 정보 없음  ⟹  배분 이득 없음
```
**첫 화살표가 깨짐.** r_b는 "교차 이웃이 ≥1개 있나"(이진)인데, w_bdry는 "이웃 중 몇 %"(연속)에서 나옴.

`MI(w_bdry; s)` 순위 (신호가 sensitive와 얼마나 연관되는가):

| 순위 | 세팅 | MI | Var | regime |
|---|---|---|---|---|
| 1 | NBA | **0.629** | 0.0438 | **saturated** |
| 2 | German | **0.350** | 0.0561 | **saturated** |
| 3 | Income | 0.191 | 0.0783 | clustered |
| 4 | Credit | 0.085 | 0.0328 | degree-skewed |
| 5–8 | Pokec ×4 | 0.011–0.039 | | clustered/mixed |
| 9 | Recidivism | **0.007** | **0.0113** | **saturated** |

- **saturated 평균 MI 0.329 vs non-saturated 0.060** — 논문 주장과 5배 차이로 정반대
- 실제로 붕괴한 건 **Recidivism 하나뿐**

**exact top-k 정당화도 반전됨.** 원고 Sec. 4.2는 "saturated에서 동점이 많다"고 하는데, 측정된 `arb_frac`은 saturated가 **최저**(German 0.000, NBA 0.000, Recidivism 0.014), 최고는 non-saturated(Pokec-z Gender 0.065, Income 0.045).

**진짜 메커니즘 가설:**

| | h | 평균 degree | rx_mean | rx_std | 결과 |
|---|---|---|---|---|---|
| Recidivism | **0.536** | 34.0 | **0.458** | 0.124 | 붕괴 |
| NBA | 0.729 | 53.7 | 0.274 | 0.190 | 강함 |
| German | 0.809 | 44.5 | 0.185 | 0.133 | 강함 |

→ **`h ≈ 0.5` × 높은 degree** 이면 r_v^×가 0.5 주변으로 몰려 분산이 붕괴.
`r_b`(경계 노드 비율)가 아님.

**결론: regime 판정 기준을 `r_b`에서 분산/MI 기반으로 교체해야 함.** 그러면 분기 3개와 컷오프 3개(`r_b<0.5`, `≥0.9`, `δ>0.2`)가 데이터에서 계산되는 단일 규칙으로 대체되고, 이론 유도와도 붙음.

---

## 4. harness/ 폴더 현황

```
harness/
├── README.md            폴더 계약 + 발견 기록
├── HANDOFF.md           이 문서
├── core/
│   ├── datasets.py 350  torch-free 로딩, .npz 캐시, provenance 가드   [검증됨]
│   ├── signals.py  279  ★ 배분 기준 12개 단일 인터페이스              [검증됨]
│   ├── allocate.py 155  score → φ(v)                                  [검증됨]
│   ├── model.py         2층 GCN (plain torch, PyG 없음, CPU 가능)     [동작 확인]
│   ├── objectives.py    L_str / L_rep / L_pred (node-weighted)        [동작 확인]
│   ├── metrics.py       acc/auc/dp/eo (numpy만)                       [동작 확인]
│   └── trainer.py       학습 루프 하나 (모든 arm 공유)                 [동작 확인]
├── experiments/e2_signal_swap.py    ★ 중심 실험
├── tests/test_core.py               38개 검사 전부 통과
├── e0_diagnostics/                  진단 산출물 (CSV + 스크립트)
└── results/e2_signal_swap.csv       파일럿 결과 48 run
```

### 설계 원칙 (깨지 않을 것)

기존 `utils/model_fairgate.py::compute_fiw_weights()`는 **345줄 한 함수**에 regime 판정 + 신호 선택 + 게이팅 + 순위 + 불확실성이 뒤엉켜 있어 **신호만 바꿔서 비교하는 게 불가능**했음. 그래서 분리:

```
signals.py  →  "누가 중요한가" (점수만)         signal(graph, state) -> np.ndarray (n,)
allocate.py →  "얼마나 세게" (점수 → 가중치)     순수 함수. regime·그래프·모델을 모름
trainer.py  →  학습. 신호가 뭔지 전혀 모름
```

**`allocate()`가 지켜야 할 불변식 3개** (테스트로 강제):
1. **예산 고정** — `mean(φ) == 1` 정확히. 안 그러면 신호를 바꿀 때 λ_fair도 같이 바뀌어 비교가 무의미.
2. **양의 하한** — `φ_min > 0`
3. **uniform으로 degrade** — 게이트 내 점수가 전부 같으면 평탄화. (원고 Sec. 4.3이 주장하는 성질. 기존 코드에선 분모 EPS 때문에 *우연히* 성립했음. 명시적으로 구현함.)

### 등록된 배분 기준 12개

| 통제 | 구조 | 모델 의존 | 기존 방법 재현 |
|---|---|---|---|
| `uniform` `random` | `w_bdry` `w_deg` `w_lhd` `w_bdry_deg_var` `w_bdry_deg_mi` | `loss` `entropy` `sigma` | `fairgb_group` `bind_influence` |

- `fairgb_group`은 테스트로 **(y,s) 4개 값만 가지며 subgroup 내 상수**임을 확인 → FairGB가 위상 정보를 안 쓴다는 게 코드로 증명됨. **논문의 구분선.**
- `bind_influence`는 현재 **1차 근사**. 정식 비교 시 `adapters/bind.py`로 실제 추정기 연결 필요.

---

## 5. E2 파일럿 결과 (48 run, CPU 17분)

`harness/results/e2_signal_swap.csv`. 4개 데이터셋 × 4신호 × 3시드.

**결론: 아직 판단 불가.** 효과가 노이즈에 묻힘.

같은 시드끼리 짝지은 비교(uniform 대비 DP 변화, 음수=도움):

```
recidivism + w_deg :  -0.0119 ± 0.0032,  3승 0패   ← 유일하게 유의
income     + w_bdry:  -0.0058 ± 0.0039,  3승 0패   ← 방향 일관, 크기 작음
나머지 10개        :  판단 불가
```

**중요**: 원고가 보고한 uniform 0.039 → adaptive 0.028 (차이 0.011)이 **지금 파일럿이 재는 크기와 동일**. 시드 3개로 안 잡힘 → **원고의 시드 5개로도 뒷받침이 어려움.**

**파일럿 설계 실수**: `PILOT_DATASETS`에서 **Credit(유일한 degree-skewed)을 빠뜨림.** `w_deg`가 빛나야 할 곳이 없었음. 그런데 정작 degree 격차가 거의 없는 Recidivism(δ_deg=0.023)에서 `w_deg`가 이김 — 논문 규칙과 반대.

---

## 6. 다음에 할 일 (순서대로)

### 즉시
1. **환경 확인** — RTX6000에 torch/numpy/pandas/scipy가 있는지. RTX5090에는 base/myenv 어디에도 없어서 격리 설치(`pip install --target`)로 작업했음.
2. **`python harness/tests/test_core.py --data`** — 9개 전부 통과해야 데이터가 온전한 것. 특히 Credit(108MB).
3. **git 초기화** — 아직 저장소가 아님. Credit 사고와 `algorithms/*copy.py` 혼동이 전부 여기서 나옴. `.gitignore`에 `data/`, `harness/.cache/`, `algorithms/adj_files/`, `outputs/`, `__pycache__` 넣을 것. 데이터는 git 대신 복원 스크립트로 기록.

### E2 본실험 (go/no-go)
4. **9개 데이터셋 전부** + **시드 10개 이상** + **짝지은 비교 기본**
   - `PILOT_DATASETS`에 Credit·Pokec 4개 추가
   - 우선 중간 규모: Credit + Pokec 4개, 4신호 × 5시드 (약 2시간)
   - **Credit에서 `w_deg`가 이기는지**가 regime 주장의 핵심
5. 결과 해석:
   - 각 고정 기준이 **자기 regime에서만** 이김 → D2 증명, 논문 성립
   - 한 기준이 어디서나 이김 → regime 주장 틀림
   - 아무것도 uniform을 못 이김 → 배분 자체가 무의미
   - **어느 쪽이든 4개월 아낌**

### 그 다음
6. `adapters/{bind,fairgb,fairsin}.py` — 외부 레포는 **절대 수정 금지**, 감싸기만. 공식 코드를 그대로 썼다는 게 검증 가능해야 함.
7. **E3 합성 sweep** — h, r_b, δ_deg 독립 변화 → 이득 곡선. 논문의 backbone 그림. **컷오프를 합성에서 정하고 동결한 뒤 실데이터에 적용** (순환 논증 회피).
8. **Phase 2 이론 유도** — CSBM/linear-GCN 1차 전개로 최적 배분 유도. 성공하면 분기 if-else가 연속 공식 하나로 대체됨. 4주 안에 안 되면 중단하고 경험적으로.
9. 베이스라인 정리:
   - **추가 필수**: BIND, ComFairGNN, FairDrop, 학습된 φ(v)(ARL식) ← "왜 규칙을 손으로 짜나"에 선제 대응
   - **부록으로 강등**: FairWalk/CrossWalk(NBA에서 AUC 0.50 = 무작위), FairEdit/EDITS(9개 중 4개 OOM)
   - **튜닝 패리티** — FairGate만 grid 48 + LHS 30을 받은 상태. 최소한 FairGB/FairGT/FairVGNN/BIND에 동일 예산 부여. **나중에 못 고침.**
10. **σ(v) 제거 검토** — 이득이 ΔDP 0.038→0.036인데 컴포넌트 1개 + head 1개 + 하이퍼파라미터 2개 + in-house 인용 의존이 딸려옴.

### 논문 구조 (실행 순서와 다름)
```
Sec 3. Motivating Observation   ← 방법론 앞에. FairGate 없이 GCN+loss 하나로.
Sec 4. Method
Sec 5. Evaluation
   E1 배분이 uniform보다 낫다        (전제)
   E2 어떤 기준이냐가 결정적이다      (D2 증명) ★ 중심
   E3 언제 통하고 안 통하는지 예측    (D3 증명) ★
   E4 SOTA 비교                     (확인, 메인 아님)
   E5 ablation
```
현재 원고는 E4가 메인이고 E2가 축소판(w_bdry 하나)으로 뒤에 있음. **뒤집을 것.**

---

## 7. 기존 코드의 알려진 결함 (harness/에서 반복 금지)

- `signal_diagnostics.py:414` — `--node-set sens`가 `get_dataset`의 2번째 반환값을 node index 배열로 취급. 실제로는 **feature 컬럼 인덱스(int)** → 평가 노드가 1개로 붕괴. (실무상 무해: 어느 데이터셋도 `sens<0`이 없어 `sens`와 `all`이 동일)
- `utils/data.py:349` — German 로더가 str 컬럼에 int 대입 → pandas 3에서 예외
- `algorithms/`에 `FairGB_alg copy.py`, `FairWalk copy.py`, `CrossWalk copy.py`가 원본과 공존. **어느 것이 제출 수치를 만들었는지 복원 불가.** harness/에는 `copy` 쌍둥이를 만들지 말 것.
- `compute_fiw_weights`에 `# jMfU Q5 rebuttal sensitivity sweep` 주석과 사후 추가 파라미터 6개. 파라미터는 설계에서 나와야지 방어에서 나오면 안 됨.
- 표를 손으로 옮기지 말 것. 원고의 수치 불일치(속도 3종, Table 11 Pokec-n Gender 복붙, App J vs Table 1)가 전부 수동 전사에서 나옴. `results/*.csv → tables/*.py → latex` 자동 생성.

---

## 8. 환경 메모

- 이전 서버(RTX5090)에는 torch/numpy가 conda 환경에 없었음. 격리 설치로 작업:
  `pip install --target=<dir> numpy pandas scipy` + `torch --index-url .../whl/cpu`
- `core/`는 torch 불필요 (numpy/pandas/scipy만). `model/objectives/trainer`만 torch 필요.
- **RTX6000 접속 시 경고**: Memory 87%, **Swap 99%**, load 28.1, `/` 89% 사용. 큰 실험 전에 `free -h`, `nvidia-smi`, `df -h` 확인할 것. Pokec(67k 노드)에서 OOM 위험.
- 파일럿 기준 CPU에서 21초/run. 9개 × 8신호 × 10시드 = 720 run이면 CPU로 하룻밤. GPU면 훨씬 빠름.
