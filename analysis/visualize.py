"""
시각화 모듈.

메인 논문용 figure:
  1. t-SNE 임베딩 시각화 (Figure 1)
  2. DICL scatter plot (Figure 2)
  3. Attention heatmap — appendix용 개별 예시 (선택)

테이블용 attention 데이터 추출:
  4. `extract_attention_ngram_csv` — 언어권별 top-k n-gram CSV 생성
     (`build_top_ngram_table.py`의 입력으로 쓰임)

Usage:
    # 전체 figure 일괄 생성
    python -m analysis.visualize all

    # t-SNE
    python -m analysis.visualize tsne \
        --centroids outputs/analysis/country_centroids.npz

    # DICL scatter
    python -m analysis.visualize scatter \
        --pairs outputs/analysis/dicl_correlation_pairs.csv \
        --summary outputs/analysis/dicl_correlation_summary.csv

    # 개별 이름 attention heatmap (appendix 용)
    python -m analysis.visualize attention \
        --model_path outputs/checkpoints/best_model.pt \
        --names "kim taeyeon" "garcia martinez" "mueller schmidt"

    # 언어권별 top-k n-gram CSV 추출 (Table 1 입력)
    python -m analysis.visualize extract_ngram \
        --model_path outputs/checkpoints/best_model.pt
"""

import os
import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 서버 환경 호환
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ─── EMNLP/ACL 제출 규격 ───
# 출처: ACL style guide (aclweb.org). 컬럼 폭 기준치.
SINGLE_COL_WIDTH = 3.3    # \begin{figure} — 1칼럼 폭 (inches)
DOUBLE_COL_WIDTH = 6.8    # \begin{figure*} — 2칼럼 폭 (inches)

# 논문 figure용 rcParams — 본문 10pt 대비 축소된 크기로 통일
# (학회에서 줄여 인쇄되어도 8pt 이상 유지되도록 설정)
PUB_RCPARAMS = {
    'font.size': 8,
    'font.family': 'serif',        # 본문 Times/CM과 어울림
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'figure.titlesize': 9,
    'axes.linewidth': 0.8,         # 축 선 얇게
    'lines.linewidth': 1.4,
    'grid.linewidth': 0.4,
    'pdf.fonttype': 42,            # TrueType 폰트 embed (Illustrator 편집 가능)
    'ps.fonttype': 42,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
}
plt.rcParams.update(PUB_RCPARAMS)


# ─── Figure 1: t-SNE 임베딩 시각화 ───

def plot_tsne(centroids_path, config_path, output_path='outputs/figures/tsne_embeddings.pdf'):
    """국가별 centroid를 t-SNE로 2D 축소 후 언어권별 색상 + convex hull로 시각화.

    재현성을 위해 t-SNE 강제 사용 (UMAP try/except 제거).
    paper 일관성 보장 — 환경에 umap-learn 설치 여부와 무관.
    """
    import yaml
    from scipy.spatial import ConvexHull
    from sklearn.manifold import TSNE

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    from data import get_loaded_df, get_lang_group_list

    df = get_loaded_df(config)
    noc_to_lang = df.drop_duplicates('NOC').set_index('NOC')['LanguageGroup'].to_dict()

    # Centroids 로드
    data = np.load(centroids_path)
    nocs = list(data.files)
    embeddings = np.array([data[noc] for noc in nocs])

    # t-SNE 차원 축소 — local cluster 분리 강화
    # · perplexity 작게 (≤15): local 구조 강조 (168개 centroid엔 작은 값이 적합)
    # · metric='cosine': character embedding의 norm 차이 무시
    # · init='pca': 안정적 초기화 (random보다 재현성 ↑)
    reducer = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(15, len(nocs) - 1),
        metric='cosine',
        init='pca',
        learning_rate='auto',
    )
    coords = reducer.fit_transform(embeddings)
    method_name = 't-SNE'

    # 언어권별 색상 (tab20 + tab20b로 39개 커버)
    lang_groups = [noc_to_lang.get(noc, 'Unknown') for noc in nocs]
    unique_langs = sorted(set(lang_groups))
    n_langs = len(unique_langs)
    color_list = []
    for i in range(n_langs):
        if i < 20:
            color_list.append(cm.tab20(i / 20))
        else:
            color_list.append(cm.tab20b((i - 20) / 20))
    color_map = {lang: color_list[i] for i, lang in enumerate(unique_langs)}

    # 플롯 (EMNLP 2-column figure* 기준)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 4.2))

    for lang in unique_langs:
        mask = np.array([lg == lang for lg in lang_groups])
        xs = coords[mask, 0]
        ys = coords[mask, 1]
        color = color_map[lang]

        # 점 (작게 — 168개 국가가 다 보여야 함)
        ax.scatter(xs, ys, c=[color], label=lang, s=16, alpha=0.85,
                   edgecolors='white', linewidths=0.3, zorder=3)

        # Convex Hull (3개 이상인 그룹만, outlier 필터 후)
        if len(xs) >= 3:
            try:
                points = np.column_stack([xs, ys])
                # Outlier 제거: 5점 이상이면 centroid에서 먼 상위 15% 제외
                # → hull이 outlier 1점에 의해 늘어나지 않음 (그룹 본체만 표현)
                if len(points) >= 5:
                    centroid = points.mean(axis=0)
                    distances = np.linalg.norm(points - centroid, axis=1)
                    keep_mask = distances <= np.percentile(distances, 85)
                    inliers = points[keep_mask]
                    if len(inliers) >= 3:
                        points = inliers
                hull = ConvexHull(points)
                hull_vertices = np.append(hull.vertices, hull.vertices[0])
                ax.fill(points[hull_vertices, 0], points[hull_vertices, 1],
                        alpha=0.10, color=color, zorder=1)
                ax.plot(points[hull_vertices, 0], points[hull_vertices, 1],
                        color=color, linewidth=0.7, alpha=0.4, zorder=2)
            except Exception:
                pass

    # 주요 국가 라벨 (EMNLP 인쇄 크기 기준 조정)
    major_countries = {'KOR', 'JPN', 'CHN', 'USA', 'GBR', 'FRA', 'GER', 'BRA',
                       'RUS', 'IND', 'MEX', 'ESP', 'ITA', 'THA', 'VIE', 'NGA',
                       'EGY', 'IRN', 'TUR', 'POL', 'SWE', 'GRE', 'ISL', 'MGL'}

    # adjustText로 라벨 자동 배치 (겹침 회피 + 화살표 자동)
    try:
        from adjustText import adjust_text
        _has_adjust = True
    except ImportError:
        print("[t-SNE] adjustText 미설치 — `pip install adjustText` 권장")
        _has_adjust = False

    label_objs = []
    for i, noc in enumerate(nocs):
        if noc in major_countries:
            t = ax.text(coords[i, 0], coords[i, 1], noc,
                        fontsize=6, fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                  alpha=0.75, edgecolor='none'),
                        zorder=4)
            label_objs.append(t)

    if _has_adjust and label_objs:
        # expand: 라벨 박스 주변 여백 (값 클수록 라벨끼리 더 떨어짐)
        # arrowprops: 옮겨진 라벨 → 원래 점 화살표 (시각적 추적)
        adjust_text(
            label_objs, ax=ax,
            expand=(1.15, 1.30),
            force_text=(0.5, 0.7),
            arrowprops=dict(arrowstyle='-', color='gray',
                            lw=0.4, alpha=0.55),
        )

    # Legend 외부 배치, 39 entries — 3열로 압축
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=5.5,
              ncol=2, framealpha=0.9, handletextpad=0.3, columnspacing=0.7,
              borderpad=0.3)
    ax.set_title(
        f'Country embedding ({method_name}), colored by language group '
        f'(168 countries, 39 groups)',
    )
    ax.set_xlabel(f'{method_name} 1')
    ax.set_ylabel(f'{method_name} 2')
    # t-SNE 좌표는 절댓값이 의미 없음 (상대 거리만 해석 가능)
    # → tick 숫자 및 grid 제거 (학계 표준 관행)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"[{method_name}] 저장: {output_path}")
    plt.close()


# ─── Figure 2: DICL Scatter Plot ───

# Figure 2에 annotation할 exemplar 국가 쌍 (NOC 기준)
#
# 선정 전략 (2026-04-24 확정, 11쌍):
#
# ① Emb 전 범위 회귀선 밀착 (8쌍) — "correlation holds across full range"
#    · Emb 0.20~0.99 전체를 cover
#    · |residual| < 0.15 (회귀선 근접)
#    · 6~7개 언어 계통 spread
#
# ② CNL 낮고 Emb 높은 above-regression outlier (3쌍) — "모델이 DICL 넘어서 학습"
#    · CNL 0.1~0.3 + Emb 0.8~0.9
#    · residual > +0.4 (회귀선 위로 크게 이탈)
#    · 세 가지 독립 메커니즘: 식민/정치적 분단/디아스포라
#
# 존재하지 않는 쌍은 자동 skip (안전 장치).
# 관련 산출: _select_quadrant_exemplars()는 4분면 자동 선정 대안으로 유지되나 현재 미사용.
DEFAULT_EXEMPLAR_PAIRS = [
    # ━━━ ① 회귀선 밀착 / 대조군 (Emb 전 범위) ━━━
    ('MGL', 'TUR', 'MGL-TUR'),    # Mongolia-Turkey      CNL=0.00 — negative control
    ('ESP', 'PAN', 'ESP-PAN'),    # Spain-Panama         CNL=0.68 — Romance on-trend
    ('GBR', 'USA', 'GBR-USA'),    # UK-USA               CNL=0.74 — Germanic on-trend
    ('ARG', 'CRC', 'ARG-CRC'),    # Argentina-Costa Rica CNL=0.90 — high-CNL anchor
    # ━━━ ② Mechanism exemplars (Table 3, EACL 개정판 §5) ━━━
    ('ESP', 'PHI', 'ESP-PHI'),    # transplant (decree)   CNL≈0.000 Emb≈0.68
    ('MOZ', 'POR', 'MOZ-POR'),    # transplant (registry) CNL=0.12  Emb≈0.92
    ('NED', 'SUR', 'NED-SUR'),    # transplant (registry) CNL=0.38  Emb≈0.72
    ('NED', 'INA', 'NED-INA'),    # no transplant         CNL≈0.001 Emb≈0.22
    ('ALG', 'FRA', 'ALG-FRA'),    # no transplant         CNL=0.014 Emb≈0.23
    ('CZE', 'SVK', 'CZE-SVK'),    # political split       CNL=0.019 Emb≈0.95
    ('CRO', 'SRB', 'CRO-SRB'),    # political split       CNL=0.015 Emb≈0.92
    ('BIH', 'CRO', 'BIH-CRO'),    # political split       CNL=0.146 Emb≈0.70 — 메타리뷰어 언급 쌍
    ('JOR', 'SYR', 'JOR-SYR'),    # shared tradition      CNL=0.004 Emb≈0.97
]

# 각 exemplar 라벨의 xytext 오프셋 (데이터 point 기준 픽셀 거리)
# 클러스터 지역은 서로 다른 방향으로 분산시켜 overlap 방지
# 오프셋 설계 원칙 (y ≤ 1.0 고정 전제):
#   · 점이 y=1.0 가까우면 라벨은 반드시 "점 아래"로 (위로 가면 잘림)
#   · 3+ 근접 cluster는 3각 배치 (right/left/down)로 분산
#   · 라벨 간 최소 간격: 수평 40px 또는 수직 15px
EXEMPLAR_LABEL_OFFSETS = {
    # 외로운 pair들 — 점 위쪽에 공간 충분, 기본 up-right
    'MGL-TUR':  (12, -20),    # below-right (y=0.13 → label y≈0.08)
    'NED-SUR':  (12, 8),
    'MOZ-POR':  (14, -8),     # right-down (JOR-SYR 라벨과 상단 밴드 분리)
    'ESP-PHI':  (12, -4),
    'BIH-CRO':  (12, 8),      # right-up (x=0.146 y=0.70, ESP-PHI와 x 분리)
    # 좌하단 근접쌍 3개 — 오른쪽 세로 밴드 분산 (ALG 0.27 / NED 0.19 / MGL 0.08)
    'NED-INA':  (14, -16),    # right-down
    'ALG-FRA':  (14, 10),     # right-up
    # 좌상단 stack (CRO 0.92 / CZE 0.95 / JOR 0.97, x<0.02) — 3방향 분산
    'CRO-SRB':  (-58, -10),   # left-down
    'CZE-SVK':  (14, -30),    # right-down 2단 (MOZ-POR 라벨과 밴드 분리)
    'JOR-SYR':  (14, 2),      # right (y=0.97, 상단 clip 방지 수평)
    # 우측 on-trend
    'ESP-PAN':  (12, -18),    # down
    'GBR-USA':  (-12, -28),   # down (y=0.98 → clip 방지)
    'ARG-CRC':  (-10, -20),   # left-down
}


def _select_quadrant_exemplars(pairs: pd.DataFrame, dicl_col: str,
                                top_k: int = 3) -> list:
    """
    4분면별 대표 쌍 자동 선정 — **CNL tertile로 분산**해서 한쪽에 몰리지 않도록.

    왜 tertile 분산:
      - CNL 분포가 right-skewed → 단순 "score 최대 top-k"는 극단 한쪽만 뽑힘
        (예: Q2의 경우 전부 CNL≈0.001, Q4는 전부 Malta)
      - 분면 내부를 top_k개 CNL 구간으로 나누고 각 구간에서 score 최고 1개씩 선정
        → x축 CNL 전체 범위에 annotation 분산

    분면별 score (구간 내부 랭킹 기준):
      Q1 top-right    CNL+Emb 합 최대 (같은 계통 corner)
      Q2 top-left     residual 최대 (회귀선 위 이탈, 예상보다 가까움)
      Q3 bottom-left  (1-CNL)+(1-Emb) 합 최대 (반대 계통 corner)
      Q4 bottom-right residual 최소 (회귀선 아래 이탈, 예상보다 멂)

    Returns: [(noc_i, noc_j, label, quadrant), ...] — top_k×4개
    """
    from scipy.stats import linregress

    x = pairs[dicl_col].values
    y = (1.0 - pairs['embedding_distance']).values
    slope, intercept, _, _, _ = linregress(x, y)
    predicted = slope * x + intercept
    residual = y - predicted

    x_med = float(np.median(x))
    y_med = float(np.median(y))

    work = pairs.copy()
    work['_x'] = x
    work['_y'] = y
    work['_residual'] = residual

    q1 = work[(work['_x'] >= x_med) & (work['_y'] >= y_med)]
    q2 = work[(work['_x'] < x_med) & (work['_y'] >= y_med)]
    q3 = work[(work['_x'] < x_med) & (work['_y'] < y_med)]
    q4 = work[(work['_x'] >= x_med) & (work['_y'] < y_med)]

    def _pick_by_tertile(sub: pd.DataFrame, score_col: str,
                          ascending: bool, n_bins: int) -> pd.DataFrame:
        """분면 내부를 CNL 기준 n_bins 구간으로 나누고 각 구간에서 score top-1 선정."""
        if len(sub) == 0:
            return sub.iloc[0:0]
        if len(sub) < n_bins:
            return sub.sort_values(score_col, ascending=ascending).head(n_bins)

        edges = np.quantile(sub['_x'], np.linspace(0, 1, n_bins + 1))
        picks = []
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                bin_df = sub[(sub['_x'] >= lo) & (sub['_x'] <= hi)]
            else:
                bin_df = sub[(sub['_x'] >= lo) & (sub['_x'] < hi)]
            if len(bin_df) > 0:
                best = bin_df.sort_values(score_col, ascending=ascending).iloc[[0]]
                picks.append(best)
        return pd.concat(picks) if picks else sub.iloc[0:0]

    q1_rep = _pick_by_tertile(
        q1.assign(_score=q1['_x'] + q1['_y']),
        score_col='_score', ascending=False, n_bins=top_k,
    )
    q2_rep = _pick_by_tertile(
        q2, score_col='_residual', ascending=False, n_bins=top_k,
    )
    q3_rep = _pick_by_tertile(
        q3.assign(_score=(1 - q3['_x']) + (1 - q3['_y'])),
        score_col='_score', ascending=False, n_bins=top_k,
    )
    q4_rep = _pick_by_tertile(
        q4, score_col='_residual', ascending=True, n_bins=top_k,
    )

    result = []
    for quad, rep in [('Q1', q1_rep), ('Q2', q2_rep),
                      ('Q3', q3_rep), ('Q4', q4_rep)]:
        for _, row in rep.iterrows():
            result.append((
                row['noc_i'], row['noc_j'],
                f"{row['noc_i']}-{row['noc_j']}",
                quad,
            ))
    return result


def _pick_dicl_column(pairs: pd.DataFrame) -> str:
    """
    pairs DataFrame에서 반드시 dicl_cnl 컬럼을 사용.

    CNL (Common Native Language index)만 허용 — LPN 등 다른 지표는
    Phase 4 분석 맥락에서 의미가 완전히 다르므로 fallback 금지.
    CNL 없으면 dicl_correlation.py 재실행 필요.
    """
    if 'dicl_cnl' in pairs.columns:
        return 'dicl_cnl'
    raise ValueError(
        f"pairs CSV에 dicl_cnl 컬럼 없음 — 실제 컬럼: {list(pairs.columns)}\n"
        f"해결: `python -m analysis.dicl_correlation --model_path <PATH>` 재실행 필요.\n"
        f"(Phase 4 분석은 CNL 기반이며 LPN 등 다른 지표로 대체 불가)"
    )


def plot_dicl_scatter(pairs_path, summary_path,
                      output_path='outputs/figures/dicl_scatter.pdf',
                      exemplar_pairs=None, n_bootstrap=300):
    """
    CNL similarity vs embedding cosine similarity 시각화.

    구성:
      · 얇은 scatter — raw 데이터 분포는 배경에 자연스럽게
      · 95% CI 밴드 (bootstrap) — 회귀선의 통계적 불확실성
      · Exemplar pair annotation — 언어학적 서사 hook
      · 축 범위 [0, 1] 고정 (유사도 정규화 범위)
    """
    from scipy.stats import linregress

    pairs = pd.read_csv(pairs_path)
    summary = pd.read_csv(summary_path)

    # primary metric 행 찾기 (cnl)
    if 'metric' in summary.columns:
        primary = summary[summary['primary'] == True]
        if len(primary) == 0:
            primary = summary.iloc[[0]]
        rho = primary['spearman_rho'].values[0]
        p_val = primary['p_value'].values[0]
        n_pairs = int(primary['n_pairs'].values[0])
    else:
        rho = summary['spearman_rho'].values[0]
        p_val = summary['p_value'].values[0]
        n_pairs = int(summary['n_pairs'].values[0])

    # DICL 지표 컬럼 — CNL 우선 선택
    dicl_col = _pick_dicl_column(pairs)

    # cosine distance → similarity 변환 (양의 상관 표시)
    emb_similarity = (1.0 - pairs['embedding_distance']).values
    dicl_values = pairs[dicl_col].values
    rho_display = -rho  # distance→similarity 부호 반전

    # ─── 그림 (EMNLP 2-column figure*) ───
    # 사이즈 확대: 논문 인쇄 시 annotation 가독성 확보 (기존 3.8 → 5.2)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, 5.2))

    # 1) 얇은 scatter — 개별 점 (hexbin 없이 raw 분포만 은은하게)
    ax.scatter(dicl_values, emb_similarity,
               s=5, alpha=0.12, color='steelblue',
               edgecolors='none', zorder=2)

    # 2) 회귀선 + bootstrap 95% CI envelope
    slope, intercept, _, _, _ = linregress(dicl_values, emb_similarity)
    x_line = np.linspace(dicl_values.min(), dicl_values.max(), 100)
    y_line = slope * x_line + intercept

    rng = np.random.RandomState(0)
    n = len(dicl_values)
    boot_ys = np.empty((n_bootstrap, len(x_line)))
    for b in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        s, i, _, _, _ = linregress(dicl_values[idx], emb_similarity[idx])
        boot_ys[b] = s * x_line + i
    y_low = np.percentile(boot_ys, 2.5, axis=0)
    y_high = np.percentile(boot_ys, 97.5, axis=0)

    ax.fill_between(x_line, y_low, y_high, alpha=0.25, color='red',
                    label='95% CI', zorder=3)
    ax.plot(x_line, y_line, 'r-', linewidth=1.5, label='Linear fit', zorder=4)

    # 3) Exemplar pair annotation
    #    기본: DEFAULT_EXEMPLAR_PAIRS (CNL≥0.5, 회귀선 밀착, 언어계통 다양성)
    #    exemplar_pairs를 명시적으로 넘기면 override (예: _select_quadrant_exemplars 결과)
    if exemplar_pairs is None:
        exemplar_pairs = DEFAULT_EXEMPLAR_PAIRS

    has_noc_cols = {'noc_i', 'noc_j'}.issubset(pairs.columns)
    if has_noc_cols:
        annotated = []
        for item in exemplar_pairs:
            # _select_quadrant_exemplars는 4-tuple (noc_i, noc_j, label, quadrant),
            # DEFAULT_EXEMPLAR_PAIRS는 3-tuple — 둘 다 지원
            a, b, label = item[0], item[1], item[2]
            mask = (((pairs['noc_i'] == a) & (pairs['noc_j'] == b)) |
                    ((pairs['noc_i'] == b) & (pairs['noc_j'] == a)))
            if not mask.any():
                continue
            row = pairs[mask].iloc[0]
            x = row[dicl_col]
            y = 1.0 - row['embedding_distance']
            ax.scatter([x], [y], s=40, facecolor='none',
                       edgecolor='black', linewidth=1.2, zorder=5)
            # 라벨별 맞춤 offset — 클러스터 overlap 방지
            dx, dy = EXEMPLAR_LABEL_OFFSETS.get(label, (8, 5))
            ax.annotate(label, (x, y), xytext=(dx, dy),
                        textcoords='offset points', fontsize=9,
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2',
                                  facecolor='white', alpha=0.9,
                                  edgecolor='gray', linewidth=0.4),
                        zorder=6,
                        # 오프셋이 크면 화살표로 연결
                        arrowprops=dict(arrowstyle='-', color='gray',
                                        lw=0.4, alpha=0.6)
                                        if max(abs(dx), abs(dy)) > 15 else None)
            annotated.append(label)
        if annotated:
            print(f"  [exemplars annotated] {annotated}")

    # 4) 통계 박스 — APA 스타일 (p < 0.001 관행)
    try:
        p_num = float(p_val)
        if p_num < 0.001:
            p_display = 'p < 0.001'
        else:
            p_display = f'p = {p_num:.3f}'
    except (ValueError, TypeError):
        # 문자열로 전달된 경우 (예: "<1e-308") — APA 관행대로 표기
        p_display = 'p < 0.001'

    # 우하단 영역 layout: Legend (위) + Stats box (아래) — 간격 최소화
    # 좌상단은 완전히 비워둠 — 데이터 영역 깨끗

    # Stats box — 맨 아래
    ax.text(0.97, 0.03,
            f'Spearman ρ = {rho_display:+.3f}\n{p_display}\nn = {n_pairs:,}',
            transform=ax.transAxes, va='bottom', ha='right', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.92, linewidth=0.5))

    ax.set_xlabel('CNL Similarity', fontsize=11)
    ax.set_ylabel('Embedding Cosine Similarity', fontsize=11)
    ax.set_xlim(0.0, 1.0)
    # y축은 음수 영역 포함 — embedding cosine similarity가 음수인 페어 (예: MGL-TUR
    # ≈ -0.13) 가 잘리지 않도록. 전체 페어의 ~38%가 sim<0이므로 y_lim 확장이 필수.
    ax.set_ylim(-0.35, 1.0)
    # Legend를 Stats box 바로 위에 — bbox_to_anchor로 정확히 배치 (간격 최소)
    ax.legend(loc='lower right', bbox_to_anchor=(0.998, 0.20),
              framealpha=0.9, fontsize=8.5, borderpad=0.4, labelspacing=0.3)
    ax.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"[Scatter] 저장: {output_path}")
    plt.close()


# ─── Figure 3: Attention Heatmap ───

def _attention_logits_from_model(model, uni, bi, tri):
    """
    Return (attn_logits [B, L, 1], padding_mask [B, L]) for either
    MultiChannelEncoder (fusion + transformer_encoder) 또는
    MultiChannelConformer (3 conformer stacks + cross-attn 또는 per-layer xattn).

    AttentionPooling.attn 직전까지 forward — pooling 가중치 추출용.
    """
    import math
    import torch
    padding_mask = (uni == 0)
    scale = math.sqrt(model.d_model)

    # Path A — MultiChannelEncoder (legacy)
    if hasattr(model, 'fusion') and hasattr(model, 'transformer_encoder'):
        e_uni = model.emb_uni(uni)
        e_bi = model.emb_bi(bi)
        e_tri = model.emb_tri(tri)
        combined = torch.cat([e_uni, e_bi, e_tri], dim=-1)
        x = model.fusion(combined) * scale
        x = model.pos_encoder(x)
        x = model.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return model.pooling.attn(x), padding_mask

    # Path B — MultiChannelConformer
    if not hasattr(model, 'conformer_uni'):
        raise ValueError(f"지원 안 되는 모델 타입: {type(model).__name__}")

    h_uni = model.emb_uni(uni) * scale
    h_bi  = model.emb_bi(bi)   * scale
    h_tri = model.emb_tri(tri) * scale

    # Optional conv stem
    if getattr(model, 'use_conv_stem', False):
        h_uni = model.stem_uni(h_uni)
        h_bi  = model.stem_bi(h_bi)
        h_tri = model.stem_tri(h_tri)

    h_uni = model.pos_encoder(h_uni)
    h_bi  = model.pos_encoder(h_bi)
    h_tri = model.pos_encoder(h_tri)

    L = h_uni.size(1)
    cat_mask = torch.cat([padding_mask] * 3, dim=1)

    # Conformer stacks (with optional per-layer cross-attn)
    if getattr(model, 'cross_attn_per_layer', False):
        for i in range(model.n_layers):
            h_uni = model.conformer_uni[i](h_uni, src_key_padding_mask=padding_mask)
            h_bi  = model.conformer_bi[i](h_bi,   src_key_padding_mask=padding_mask)
            h_tri = model.conformer_tri[i](h_tri, src_key_padding_mask=padding_mask)
            cat = torch.cat([h_uni, h_bi, h_tri], dim=1)
            cross_out, _ = model.cross_attns[i](
                cat, cat, cat, key_padding_mask=cat_mask, need_weights=False)
            cat = model.cross_norms[i](cat + cross_out)
            h_uni = cat[:, :L, :]
            h_bi  = cat[:, L:2 * L, :]
            h_tri = cat[:, 2 * L:, :]
    else:
        for blk in model.conformer_uni: h_uni = blk(h_uni, src_key_padding_mask=padding_mask)
        for blk in model.conformer_bi:  h_bi  = blk(h_bi,  src_key_padding_mask=padding_mask)
        for blk in model.conformer_tri: h_tri = blk(h_tri, src_key_padding_mask=padding_mask)

    # Output branch — channel_ensemble_head vs cross-attn fusion
    if getattr(model, 'channel_ensemble_head', False):
        # 3 채널 attention 평균 (channel별 독립 pool 평균)
        if model.pool_uni is None:
            raise ValueError("channel_ensemble + non-attention pool 조합은 attention 추출 불가")
        attn_logits = (model.pool_uni.attn(h_uni)
                       + model.pool_bi.attn(h_bi)
                       + model.pool_tri.attn(h_tri)) / 3.0
        return attn_logits, padding_mask

    # 기본: 마지막 cross-attn 후 단일 pooling
    combined = torch.cat([h_uni, h_bi, h_tri], dim=1)
    cross_out, _ = model.cross_attn(
        combined, combined, combined,
        key_padding_mask=cat_mask, need_weights=False)
    h_fused = model.cross_norm(h_uni + cross_out[:, :L, :])
    return model.pooling.attn(h_fused), padding_mask


def plot_attention_heatmap(config_path, model_path, names,
                           output_path='outputs/figures/attention_heatmap.pdf'):
    """
    이름 입력에 대한 AttentionPooling 가중치를 heatmap으로 시각화.
    모델이 이름의 어떤 n-gram에 집중하는지 보여준다.
    """
    import yaml
    import torch

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = config['experiment']['device']
    if not torch.cuda.is_available():
        device = 'cpu'

    from data import get_vocab_size, get_ngram_dicts, get_country_idx, get_lang_group_idx
    from models import get_model
    from utils.util import replace_disallowed_chars, text_to_unigram_indices, \
        text_to_bigram_indices, text_to_trigram_indices

    vocab_size = get_vocab_size(config)
    ngram_dicts = get_ngram_dicts(config['data']['ngram'])
    max_length = config['data']['max_length']

    # 모델 로드
    model = get_model(
        config, vocab_size,
        n_classes=config['model']['num_classes'],
        n_lang_groups=config['model']['num_lang_groups'],
    )
    _state = torch.load(model_path, map_location=device)
    _missing, _unexpected = model.load_state_dict(_state, strict=False)
    if _missing or _unexpected:
        print(f"  [load_state_dict] strict=False  missing={len(_missing)}  unexpected={len(_unexpected)}")
    model = model.to(device)
    model.eval()

    # EMNLP 2-column figure*, 이름 수에 따라 높이 조정
    fig, axes = plt.subplots(len(names), 1,
                              figsize=(DOUBLE_COL_WIDTH, 0.9 * len(names) + 0.3))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        clean_name = replace_disallowed_chars(name)

        # 입력 생성
        uni = torch.LongTensor(
            text_to_unigram_indices(clean_name, ngram_dicts['uni'], max_length)
        ).unsqueeze(0).to(device)
        bi = torch.LongTensor(
            text_to_bigram_indices(clean_name, ngram_dicts['bi'], max_length)
        ).unsqueeze(0).to(device)
        tri = torch.LongTensor(
            text_to_trigram_indices(clean_name, ngram_dicts['tri'], max_length)
        ).unsqueeze(0).to(device)

        # Forward — model-type-aware (MultiChannelEncoder vs MultiChannelConformer)
        with torch.no_grad():
            attn_weights, padding_mask = _attention_logits_from_model(model, uni, bi, tri)
            # Pooling softmax (padding 위치는 -1e4로 mask)
            attn_weights = attn_weights.masked_fill(padding_mask.unsqueeze(-1), -1e4)
            attn_weights = torch.softmax(attn_weights, dim=1)
            attn_weights = attn_weights.squeeze().cpu().numpy()  # [L]

        # 유효 길이만
        name_len = min(len(clean_name), max_length)
        attn = attn_weights[:name_len]

        # 각 position의 n-gram 라벨 생성
        chars = list(clean_name)
        labels = []
        for j in range(name_len):
            u = chars[j] if j < len(chars) else ''
            b = clean_name[j:j+2] if j+1 < len(clean_name) else chars[j]
            t = clean_name[j:j+3] if j+2 < len(clean_name) else clean_name[j:len(clean_name)]
            labels.append(f'{u}\n{b}\n{t}')

        # Heatmap
        attn_2d = attn.reshape(1, -1)
        ax.imshow(attn_2d, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(name_len))
        ax.set_xticklabels(labels, fontsize=5.5, family='monospace')
        ax.set_yticks([])
        ax.set_title(f'"{name}"', fontsize=8, fontweight='bold', loc='left')

        # 각 셀에 attention 값 표시 (작은 figure 기준 크기 조정)
        for j in range(name_len):
            ax.text(j, 0, f'{attn[j]:.2f}', ha='center', va='center', fontsize=5.5)

    plt.suptitle('Attention pooling weights (per-position, uni/bi/tri stacked)',
                 fontsize=9, y=1.00)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    print(f"[Attention] 저장: {output_path}")
    plt.close()


# ─── Figure 4/5: 언어권별 attention 분석 (Aggregate) ───

def _extract_test_attention(config_path, model_path,
                            max_samples_per_group=150, seed=42):
    """
    Test set에서 언어권별 stratified sampling → attention weight + 원본 이름 수집.

    왜 stratified sampling인가:
    - 언어권 분포가 극도로 불균형 (예: English 수천 vs 소수 언어권 수십).
    - 단순 랜덤 샘플링 시 소수 언어권이 거의 안 뽑혀 집계가 불안정.

    Returns:
        pd.DataFrame: columns = ['name', 'lang_group', 'attn', 'length']
          - attn: np.ndarray shape=[length] (softmax-normalized)
          - length: min(len(name), max_length)
    """
    import yaml
    import math
    import torch

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = config['experiment']['device']
    if not torch.cuda.is_available():
        device = 'cpu'

    from data import prepare_data, get_vocab_size
    from models import get_model

    # prepare_data는 augmentation 포함 전체 파이프라인 — test split은 결정적이므로 재현됨
    _, _, test_loader, _, _ = prepare_data(config)
    test_ds = test_loader.dataset

    vocab_size = get_vocab_size(config)
    model = get_model(
        config, vocab_size,
        n_classes=config['model']['num_classes'],
        n_lang_groups=config['model']['num_lang_groups'],
    )
    _state = torch.load(model_path, map_location=device)
    _missing, _unexpected = model.load_state_dict(_state, strict=False)
    if _missing or _unexpected:
        print(f"  [load_state_dict] strict=False  missing={len(_missing)}  unexpected={len(_unexpected)}")
    model = model.to(device).eval()

    names_all = test_ds.texts.reset_index(drop=True)
    langs_all = test_ds.lang_labels.reset_index(drop=True)
    max_length = config['data']['max_length']

    # 언어권별 stratified sampling
    rng = np.random.RandomState(seed)
    chosen_indices = []
    for lang in sorted(langs_all.unique()):
        idxs = np.where(langs_all.values == lang)[0]
        n_sample = min(len(idxs), max_samples_per_group)
        chosen = rng.choice(idxs, size=n_sample, replace=False)
        chosen_indices.extend(chosen.tolist())
    chosen_indices = sorted(chosen_indices)

    # 배치 단위 forward — 모델 성능(eval mode) 기준 attention 추출
    BATCH = 128
    results = []
    with torch.no_grad():
        for start in range(0, len(chosen_indices), BATCH):
            batch_idxs = chosen_indices[start:start + BATCH]
            uni_batch, bi_batch, tri_batch = [], [], []
            for idx in batch_idxs:
                u, b, t, _, _ = test_ds[idx]
                uni_batch.append(u)
                bi_batch.append(b)
                tri_batch.append(t)
            uni = torch.stack(uni_batch).to(device)
            bi = torch.stack(bi_batch).to(device)
            tri = torch.stack(tri_batch).to(device)

            # model-type-aware forward (MultiChannelEncoder | MultiChannelConformer)
            attn, padding_mask = _attention_logits_from_model(model, uni, bi, tri)
            attn = attn.masked_fill(padding_mask.unsqueeze(-1), -1e4)
            attn = torch.softmax(attn, dim=1).squeeze(-1).cpu().numpy()  # [B, L]

            for i, idx in enumerate(batch_idxs):
                name = names_all.iloc[idx]
                length = min(len(name), max_length)
                results.append({
                    'name': name,
                    'lang_group': langs_all.iloc[idx],
                    'attn': attn[i][:length].astype(np.float32),
                    'length': length,
                })

    return pd.DataFrame(results)


def export_top_ngram_csv(attn_df, ngram_type='tri', min_freq=5,
                          csv_path='outputs/figures/top_ngrams_by_lang_group.csv'):
    """
    언어권별 attention-weighted n-gram 집계 → CSV 저장.

    지표: n-gram이 언어권 내에서 등장할 때의 **평균 attention weight**.
      - min_freq 미만 등장 n-gram은 제외 (노이즈 필터).
      - coverage = (해당 n-gram 포함 이름 수) / (언어권 샘플 수).
        → 희귀 n-gram이 우연히 고득점한 케이스 구분 가능.

    CSV 컬럼: lang_group, ngram, avg_attn, coverage, freq
    build_top_ngram_table.py의 'attention' 모드 입력으로 사용.
    """
    from collections import defaultdict

    n_map = {'uni': 1, 'bi': 2, 'tri': 3}
    if ngram_type not in n_map:
        raise ValueError(f"ngram_type은 uni/bi/tri 중 하나: {ngram_type}")
    n = n_map[ngram_type]

    lang_ng_attn_sum = defaultdict(lambda: defaultdict(float))
    lang_ng_freq = defaultdict(lambda: defaultdict(int))
    lang_sample_count = attn_df['lang_group'].value_counts().to_dict()

    for _, row in attn_df.iterrows():
        name = row['name']
        attn = row['attn']
        length = row['length']
        lang = row['lang_group']
        for j in range(length):
            if j + n > len(name):
                break
            ng = name[j:j + n]
            # 공백 포함 n-gram은 성/이름 경계에 걸친 것 — 형태학적 단서 아님
            if ' ' in ng or len(ng) < n:
                continue
            lang_ng_attn_sum[lang][ng] += float(attn[j])
            lang_ng_freq[lang][ng] += 1

    all_rows = []
    for lang, ng_attn in lang_ng_attn_sum.items():
        for ng, total_attn in ng_attn.items():
            freq = lang_ng_freq[lang][ng]
            if freq < min_freq:
                continue
            avg_attn = total_attn / freq
            coverage = freq / lang_sample_count[lang]
            all_rows.append({
                'lang_group': lang,
                'ngram': ng,
                'avg_attn': avg_attn,
                'coverage': coverage,
                'freq': freq,
            })

    results_df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    results_df.sort_values(['lang_group', 'avg_attn'],
                           ascending=[True, False]).to_csv(csv_path, index=False)
    print(f"[{ngram_type}-gram] CSV 저장: {csv_path} "
          f"(언어권 {results_df['lang_group'].nunique()}개, n-gram {len(results_df)}행)")
    return results_df


def run_attention_extraction(config_path='configs/base_config.yaml', model_path=None,
                              max_samples_per_group=150,
                              output_dir='outputs/figures'):
    """
    Test set에서 attention 추출 → 언어권별 bi-gram/tri-gram CSV 생성.

    이 CSV를 build_top_ngram_table.py가 읽어서 논문용 Table을 생성한다.
    (별도 figure 생성 없음 — Table이 figure 대체)
    """
    import yaml
    if model_path is None:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        model_path = cfg.get('inference', {}).get('model_path',
                                                  'outputs/checkpoints/best_model.pt')

    print("\n[Attention 추출] test set forward pass 중...")
    attn_df = _extract_test_attention(config_path, model_path, max_samples_per_group)
    print(f"  수집: {len(attn_df)}건, 언어권 {attn_df['lang_group'].nunique()}개")

    os.makedirs(output_dir, exist_ok=True)
    export_top_ngram_csv(
        attn_df, ngram_type='bi',
        csv_path=os.path.join(output_dir, 'top_bigrams_by_lang_group.csv'),
    )
    export_top_ngram_csv(
        attn_df, ngram_type='tri',
        csv_path=os.path.join(output_dir, 'top_trigrams_by_lang_group.csv'),
    )
    return attn_df


# ─── 전체 시각화 일괄 실행 ───

DEFAULT_ATTENTION_NAMES = [
    "kim taeyeon", "garcia martinez", "mueller schmidt",
    "tanaka yuki", "silva santos", "olsen hansen",
]


def plot_all(config_path='configs/base_config.yaml',
             centroids_path='outputs/analysis/ours_single_seed/country_centroids.npz',
             pairs_path='outputs/analysis/ours_single_seed/dicl_correlation_pairs.csv',
             summary_path='outputs/analysis/ours_single_seed/dicl_correlation_summary.csv',
             model_path=None,
             names=None,
             output_dir='outputs/figures'):
    """세 시각화를 순차 실행. 개별 실패는 다른 단계 진행을 막지 않는다."""
    import traceback

    # 모델 경로 기본값은 config의 inference.model_path
    if model_path is None:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        model_path = cfg.get('inference', {}).get('model_path',
                                                  'outputs/checkpoints/best_model.pt')
    if names is None:
        names = DEFAULT_ATTENTION_NAMES

    os.makedirs(output_dir, exist_ok=True)
    results = {}  # 단계별 성공/실패 기록

    # 1) t-SNE
    print("\n[1/3] t-SNE 임베딩 시각화")
    try:
        if not os.path.exists(centroids_path):
            raise FileNotFoundError(f"centroids 파일 없음: {centroids_path}")
        plot_tsne(centroids_path, config_path,
                  os.path.join(output_dir, 'tsne_embeddings.pdf'))
        results['tsne'] = 'ok'
    except Exception as e:
        results['tsne'] = f'skip: {e}'
        print(f"  → 건너뜀: {e}")

    # 2) DICL scatter
    print("\n[2/3] DICL scatter plot")
    try:
        for p in (pairs_path, summary_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f"DICL 파일 없음: {p}")
        plot_dicl_scatter(pairs_path, summary_path,
                          os.path.join(output_dir, 'dicl_scatter.pdf'))
        results['scatter'] = 'ok'
    except Exception as e:
        results['scatter'] = f'skip: {e}'
        print(f"  → 건너뜀: {e}")

    # 3) Attention heatmap (개별 예시 — appendix 용)
    print("\n[3/3] Attention heatmap (예시 이름)")
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 체크포인트 없음: {model_path}")
        plot_attention_heatmap(config_path, model_path, names,
                               os.path.join(output_dir, 'attention_heatmap.pdf'))
        results['attention'] = 'ok'
    except Exception as e:
        results['attention'] = f'skip: {e}'
        traceback.print_exc()

    # 요약
    print("\n" + "=" * 50)
    print("시각화 결과 요약")
    print("=" * 50)
    for step, status in results.items():
        mark = "✓" if status == 'ok' else "✗"
        print(f"  {mark} {step}: {status}")
    print(f"\n출력 위치: {output_dir}/")
    return results


# ─── CLI ───

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualization")
    subparsers = parser.add_subparsers(dest='command')

    # All (일괄 실행)
    p_all = subparsers.add_parser('all', help='세 시각화를 한 번에 생성')
    p_all.add_argument('--config', type=str, default='configs/base_config.yaml')
    p_all.add_argument('--centroids', type=str,
                       default='outputs/analysis/ours_single_seed/country_centroids.npz')
    p_all.add_argument('--pairs', type=str,
                       default='outputs/analysis/ours_single_seed/dicl_correlation_pairs.csv')
    p_all.add_argument('--summary', type=str,
                       default='outputs/analysis/ours_single_seed/dicl_correlation_summary.csv')
    p_all.add_argument('--model_path', type=str, default=None,
                       help='미지정 시 config.inference.model_path 사용')
    p_all.add_argument('--names', nargs='+', default=None,
                       help='attention heatmap에 사용할 이름 목록')
    p_all.add_argument('--output_dir', type=str, default='outputs/figures')

    # t-SNE
    p_tsne = subparsers.add_parser('tsne')
    p_tsne.add_argument('--centroids', type=str, required=True)
    p_tsne.add_argument('--config', type=str, default='configs/base_config.yaml')
    p_tsne.add_argument('--output', type=str, default='outputs/figures/tsne_embeddings.pdf')

    # Scatter
    p_scatter = subparsers.add_parser('scatter')
    p_scatter.add_argument('--pairs', type=str, required=True)
    p_scatter.add_argument('--summary', type=str, required=True)
    p_scatter.add_argument('--output', type=str, default='outputs/figures/dicl_scatter.pdf')

    # Attention (예시 이름 heatmap)
    p_attn = subparsers.add_parser('attention')
    p_attn.add_argument('--config', type=str, default='configs/base_config.yaml')
    p_attn.add_argument('--model_path', type=str, required=True)
    p_attn.add_argument('--names', nargs='+', required=True)
    p_attn.add_argument('--output', type=str, default='outputs/figures/attention_heatmap.pdf')

    # 언어권별 top-k n-gram CSV 추출 (Table 1 입력)
    p_ngram = subparsers.add_parser('extract_ngram',
                                    help='언어권별 top-k n-gram CSV 추출 (build_top_ngram_table.py 입력)')
    p_ngram.add_argument('--config', type=str, default='configs/base_config.yaml')
    p_ngram.add_argument('--model_path', type=str, default=None,
                         help='미지정 시 config.inference.model_path 사용')
    p_ngram.add_argument('--samples_per_group', type=int, default=150,
                         help='언어권별 stratified 샘플 수 (기본 150)')
    p_ngram.add_argument('--output_dir', type=str, default='outputs/figures')

    args = parser.parse_args()

    if args.command == 'all':
        plot_all(
            config_path=args.config,
            centroids_path=args.centroids,
            pairs_path=args.pairs,
            summary_path=args.summary,
            model_path=args.model_path,
            names=args.names,
            output_dir=args.output_dir,
        )
    elif args.command == 'tsne':
        plot_tsne(args.centroids, args.config, args.output)
    elif args.command == 'scatter':
        plot_dicl_scatter(args.pairs, args.summary, args.output)
    elif args.command == 'attention':
        plot_attention_heatmap(args.config, args.model_path, args.names, args.output)
    elif args.command == 'extract_ngram':
        run_attention_extraction(
            config_path=args.config,
            model_path=args.model_path,
            max_samples_per_group=args.samples_per_group,
            output_dir=args.output_dir,
        )
    else:
        parser.print_help()
