---
name: nto2b-eda-validation
description: Проведение EDA, настройка без-утечковой GroupKFold валидации и диагностика ошибок по срезам (level, answer_format, retrieval vs reasoning) для кейса НТО ИИ 2B.
---

# NTO 2B — EDA, GroupKFold и диагностика ошибок

## Когда использовать
- Перед любым изменением пайплайна (нужен бейзлайн-замер и диагностика узкого места).
- После каждого эксперимента — для сравнения OOF-метрики по `light / medium / heavy` и по форматам ответов.

## Обязательные правила валидации
1. Всегда используй `train_groups.csv` (`GroupKFold(n_splits=5)`). Проверяй, что пересечение `group` между train и val пустое.
2. Считай метрику строго через официальный `score.py` (или `score (2).py`), учитывая деление веса для вопросов одной истории с разными `asked_at`.
3. Разделяй ошибки на 4 корзины — без этого нельзя понять, что улучшать:
   - **Retrieval Recall@K miss**: ни один золотой `post_id` (или меньше 2 для multi-hop) не попал в топ-K кандидатов.
   - **Rerank / Context miss**: золотые посты были в топ-100, но не вошли в топ-N контекста LLM или топ-5 `evidence`.
   - **Reasoning / Math error**: все нужные посты были в контексте LLM, но ответ неверный (особенно разница дат, подсчёт переносов, исторический статус).
   - **Format / Abstention error**: неверные единицы (`units.py`), неверный `promised` vs `done` vs `no_info`, лишний или пропущенный `skip`.

## Диагностическая таблица (выводить после каждого прогона)
Печатай сводку по:
1. Общему `S` и вкладу каждого `level` (`light` 19.1% веса, `medium` 33.7%, `heavy` 47.3%).
2. Раздельно `mean(a_q)` и `mean(e_q)` (AP@5) по `level` и `answer_format` (`number`, `yesno`, `choice`, `status`, `date`).
3. Recall@5 / Recall@20 / Recall@50 / Recall@100 на этапе retrieval и после reranking.
4. Confusion matrix для спец-классов: `done / promised / postponed / cancelled / no_info / skip`.
5. Топ-15 самых «дорогих» ошибок (отсортированных по `w_q * (1.0 - s_q)`), с выводом вопроса, `asked_at`, предсказания, золота и того, были ли золотые посты в контексте.

## Быстрые проверки целостности
- `assert all(post_date[pid] <= asked_at for pid in pred_evidence)` — 0 утечек из будущего.
- `assert len(pred_evidence) <= 5` и нет дубликатов `post_id`.
- Проверка `examples_gold (2).jsonl` по подтипам `A1–A5, B1–B3, C1–C6, D1–D2, S1` как регрессионного набора.
