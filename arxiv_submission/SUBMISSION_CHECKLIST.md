# arXiv Pre-Submission Checklist

## Paper Quality
- [x] Abstract ≤ 1920 characters (check abstract.txt)
- [x] All sections present: Abstract, Intro, Related Work, Method, Results, Discussion, Limitations, Conclusion
- [x] All figures referenced in text
- [x] All references in bibliography
- [ ] No broken cross-references (verify after PDF generation)
- [x] Statistical claims backed by specific numbers (CI, p-values, R²)
- [x] 6 limitations explicitly listed
- [ ] Acknowledgments section (optional but professional)

## Data & Reproducibility
- [x] All dataset files in dataset/ folder
- [x] Code fully documented with usage examples
- [x] Replication guide covers full pipeline end-to-end
- [x] Contamination report included (contamination_report.json)
- [x] Random seeds documented (bootstrap seed=42)
- [x] Model versions recorded in result JSON files
- [x] Cost breakdown included (Appendix B)

## Closing Weak Spots
- [x] WEAK SPOT 1 (No peer review) → LessWrong post + r/MachineLearning post planned
- [x] WEAK SPOT 2 (No independent replication) → REPLICATION_GUIDE.md + open data
- [x] Statistical rigor: bootstrap 95% CI for ALL key metrics
- [x] McFadden R² for sigmoid goodness-of-fit
- [x] Hosmer-Lemeshow calibration test
- [x] Mann-Whitney U pairwise significance matrix
- [x] Extended probe range (k = 1.25, 1.5) captures 2 additional transitions
- [x] 3+ new model clients ready (Claude, DeepSeek, Mistral) — run immediately after API keys acquired

## arXiv Specific
- [ ] PDF generated from paper_neurips.md
- [ ] PDF reviewed for formatting issues
- [ ] All figures: 300 DPI minimum, PNG or PDF format
- [ ] Figure captions descriptive (not just "Figure 1")
- [ ] arXiv account created (arxiv.org/register)
- [ ] Submission category confirmed: cs.AI + cs.LG + stat.ML
- [ ] Abstract paste-tested into arXiv form (check no LaTeX errors)

## Post-Submission
- [ ] arXiv ID received
- [ ] Update abstract.txt with final arXiv ID
- [ ] Share on LessWrong (target: within 24h of arXiv ID)
- [ ] Share on Twitter/X (tag @_akhaliq)
- [ ] Upload dataset to HuggingFace Hub
- [ ] Submit to papers.huggingface.co
- [ ] Open Philanthropy application (see grant/)

## Timeline (26 April 2026)
```
TODAY:    arXiv account + upload → timestamp locked as Apr 26, 2026
DAY 2-3:  Run DeepSeek-V3 + Claude Haiku (< $0.02 total)
DAY 4-7:  LessWrong post + community review (closes WEAK SPOT 1)
DAY 14:   Updated analysis with 2+ new models (closes WEAK SPOT 2 partially)
MAY:      NeurIPS 2026 submission (deadline ~late May)
```
