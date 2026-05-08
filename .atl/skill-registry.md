# Skill Registry - Proyecto Final Analitica

## Project Standards
- **Stack**: Python, Streamlit, Scikit-learn
- **Architecture**: Data Science Pipeline (Loading -> Preprocessing -> Training -> Inference)
- **UI**: Streamlit with custom CSS (Baloo 2 / Nunito fonts)
- **Testing**: No test runner detected.

## Registered Skills
| Skill | Trigger | Scope |
|-------|---------|-------|
| sdd-explore | /sdd-explore | SDD Exploration |
| sdd-propose | /sdd-propose | SDD Proposal |
| sdd-spec | /sdd-spec | SDD Specification |
| sdd-design | /sdd-design | SDD Design |
| sdd-tasks | /sdd-tasks | SDD Task breakdown |
| sdd-apply | /sdd-apply | SDD Implementation |
| sdd-verify | /sdd-verify | SDD Verification |
| sdd-archive | /sdd-archive | SDD Archiving |
| work-unit-commits | preparing commits | Git Conventions |
| cognitive-doc-design | writing docs | Documentation Style |

## Compact Rules
### Python / Data Science
- Use `joblib` for model persistence.
- Follow PEP 8 standards.
- Keep training and inference logic separated (train_drugs.py vs app.py).
- Use `st.cache_resource` or `st.cache_data` for heavy operations in Streamlit.

### UI Standards
- Use existing CSS tokens in `app.py`.
- Maintain the "Baloo 2" and "Nunito" typography.
- Use glassmorphism and soft gradients as established.
