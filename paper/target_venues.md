# Target Venues — recommendations for *ScoreCompose-AI*

The paper sits at the intersection of (a) symbolic music generation, (b)
human–AI co-creative interfaces, and (c) systems-level optimization
(incremental decoding). That intersection is unusual; it gives several
viable venues, ranked below by fit.

## Tier 1 — best fit

### TISMIR — *Transactions of the International Society for Music Information Retrieval*
- **Why**: gold-standard journal for symbolic music + ML; open access; reviewers expect reproducible artifacts. Edit-aware decoding is exactly the kind of methodological contribution they value.
- **Format**: long-form, archival.
- **Acceptance rate**: ~25–35%.
- **URL**: https://transactions.ismir.net

### ISMIR (conference proceedings)
- **Why**: the field's central venue; proceedings are peer-reviewed and citable. Strong for the **system + algorithm + evaluation** package the paper presents.
- **Format**: 6–8 pp short paper.
- **Deadline**: typically March–April for fall conference.
- **URL**: https://ismir.net

### NIME — *New Interfaces for Musical Expression*
- **Why**: the real-time editor + sub-50 ms latency story is squarely in NIME's lane. Reviewers care about interaction quality and live use.
- **Format**: 4–8 pp + demo.
- **URL**: https://nime.org

## Tier 2 — strong fit

### Computer Music Journal (MIT Press)
- **Why**: prestigious, broader scope (composition, computation, aesthetics). Good fit if the paper is expanded with more discussion of compositional workflow and a small user study.
- **Format**: long-form.
- **URL**: https://direct.mit.edu/comj

### Journal of New Music Research (Taylor & Francis)
- **Why**: long-running journal that covers symbolic representation, algorithmic composition, and HCI angles. Receptive to interactive systems.
- **URL**: https://www.tandfonline.com/journals/nnmr20

### Sound and Music Computing (SMC) — conference + journal special issues
- **Why**: European counterpart to ISMIR/NIME with a friendly review culture for systems papers.
- **URL**: https://smcnetwork.org

### Organised Sound (Cambridge Univ. Press)
- **Why**: stronger humanities/electroacoustic angle but accepts technical pieces with cultural framing. Fits authors who already publish across musicology and CS.
- **URL**: https://www.cambridge.org/core/journals/organised-sound

## Tier 3 — adjacent venues worth considering

### ACM CHI / IUI / Creativity & Cognition
- **Why**: if the contribution is reframed as an HCI primitive (low-latency edits in generative co-creation), CHI Late-Breaking Work or IUI is a plausible target. Requires a small user study.

### NeurIPS / ICML "Machine Learning for Creativity and Design" workshop
- **Why**: good early-stage venue for the algorithmic contribution alone (incremental decoding aligned to user-visible structure), with cross-pollination from the core ML community.

### Korean venues (이미 한국 학회 활동 중이신 경우)
- **한국컴퓨터음악학회 (KEAMS)** — 학회지 *전자음악연구* / 학술대회. 한국어 출간 가능.
- **Journal of the Acoustical Society of Korea (한국음향학회지)** — 음향·신호처리 쪽 색채가 강하지만 음악 정보 처리도 수용.
- **한국정보과학회 ACK 또는 KCC** — 시스템·HCI 부문에 제출 가능.

## Suggested submission strategy

1. **First submission**: ISMIR (short paper) — accepts well-scoped systems contributions, fast turnaround.
2. **If accepted at ISMIR**: extend to TISMIR (factor ~2× length, add formal latency analysis + user study).
3. **If rejected at ISMIR**: send the extended version directly to TISMIR or to JNMR.
4. **Parallel demo track**: NIME demo paper for the live editor — these are often accepted alongside a primary archival submission.

## Reviewer concerns to pre-empt

- *Novelty vs. speculative decoding.* — Address explicitly: the contribution is not faster *generation*; it is alignment of cache invalidation with user-visible structural units (notes), which is an interaction-level claim, not a pure-throughput one.
- *Lack of a large user study.* — Add at minimum a 6–10 participant within-subjects study comparing the edit-aware path to a cold-decode baseline; report task completion time and SUS.
- *Single-instrument limitation.* — Explicitly listed as a limitation; sketch the multi-track extension (interleaved tracks à la Compound Word Transformer).
