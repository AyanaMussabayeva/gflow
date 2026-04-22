# Sequential Decision Project: v2 Pipeline

Этот репозиторий теперь содержит новый воспроизводимый `pkg`-pipeline для экспериментов по GFlowNet-guided BO.

Старая версия результатов остается в `results/` и `project_report/img/`.
Новая версия ничего не перезаписывает и пишет артефакты отдельно:

- `artifacts_v2/` — сырые результаты запусков и suite manifests
- `report_assets_v2/` — готовые картинки, таблицы и report snippets

## Environment

Минимальная установка:

```bash
pip install -r requirements.txt
```

Быстрая sanity-проверка:

```bash
python -m compileall pkg tests scripts
```

## Основные скрипты

### 1. Основной suite

Smoke-run:

```bash
python scripts/run_v2_suite.py --suite main_v2 --profile smoke
```

Полный запуск:

```bash
python scripts/run_v2_suite.py --suite main_v2 --profile full
```

Что получится:

- по одному artifact run на каждый benchmark в `artifacts_v2/main_v2/<benchmark>/<timestamp>/`
- suite manifest в `artifacts_v2/_suites/main_v2/<timestamp>/manifest.json`

### 2. Reward protocol ablation

```bash
python scripts/run_reward_protocol_ablation_v2.py --profile smoke
```

Этот suite запускает:

- `reward_protocol_softplus_scaled_v2`
- `reward_protocol_zscore_v2`
- `reward_protocol_rank_v2`

### 3. Pool ablation

```bash
python scripts/run_v2_suite.py --suite pool_ablation_v2 --profile smoke
```

Этот suite запускает:

- `pool_shared_v2`
- `pool_fresh_v2`

### 4. Finetune ablation

```bash
python scripts/run_v2_suite.py --suite finetune_ablation_v2 --profile smoke
```

Этот suite запускает:

- `finetune_continual_v2`
- `finetune_restart_v2`

## Экспорт report-ready артефактов

### Main v2

```bash
python scripts/export_report_assets_v2.py --spec main_v2 --suite-name main_v2
```

Результат:

- `report_assets_v2/main_v2/<timestamp>/img/`
- `report_assets_v2/main_v2/<timestamp>/comparison/`
- `report_assets_v2/main_v2/<timestamp>/tables/`
- `report_assets_v2/main_v2/<timestamp>/snippets/`

### Reward protocol ablation

```bash
python scripts/export_reward_protocol_ablation_v2.py
```

Результат:

- `report_assets_v2/reward_protocol_ablation_v2/<timestamp>/<spec_name>/...`

Для каждого reward protocol будет свой набор:

- `img/`
- `comparison/`
- `tables/summary_metrics.csv`
- `tables/summary_metrics.md`
- `snippets/report_snippet.md`

## На что смотреть после запуска

### 1. Главные summary-файлы

Внутри каждого artifact run:

- `spec.json` — конфиг запуска
- `summary.json` — агрегированные результаты по методам
- `trial_metrics.csv` — по-seed метрики

Внутри report assets:

- `tables/summary_metrics.md`
- `snippets/report_snippet.md`

### 2. Ключевые метрики

Смотри в первую очередь на:

- `final_regret_mean`
- `regret_gain_of_gfn`
- `slowdown_factor`
- `floor_reward_fraction_mean`
- `step_reward_std_mean`
- `step_improvement_std_mean`
- `same_mask_repeat_std_mean`
- `proxy_actual_gain_corr_mean`

Интерпретация:

- высокий `regret_gain_of_gfn` — GFN лучше baseline
- высокий `slowdown_factor` — GFN дороже по compute
- высокий `floor_reward_fraction_mean` — reward деградирует к почти константе
- высокий `same_mask_repeat_std_mean` — reward/evaluation pipeline шумный
- низкий или отрицательный `proxy_actual_gain_corr_mean` — proxy reward плохо согласован с реальным improvement

### 3. Главные картинки

Ищи в `report_assets_v2/.../img/`:

- `*_regret_comparison.png`
- `*_random_diagnostics.png`
- `*_gfn_diagnostics.png`
- `*_reward_hist.png`
- `gfn_reward_hist.png`

Что они показывают:

- `regret_comparison` — итоговое сравнение random vs GFN
- `*_diagnostics` — как proxy score связан с фактическими BO query outcomes
- `reward_hist` — не схлопывается ли reward distribution

### 4. Сравнение со старой реализацией

Ищи в `report_assets_v2/.../comparison/`:

- `*_legacy_vs_v2_regret.png`

Это quickest check на вопрос:

`v2` действительно стал стабильнее или нет по сравнению с legacy pipeline.

## Рекомендуемый workflow

Если нужен быстрый исследовательский цикл:

1. Запустить `main_v2` в `smoke`
2. Проверить `summary.json` и `trial_metrics.csv`
3. Экспортировать report assets
4. Посмотреть `regret_comparison`, `reward_hist`, `legacy_vs_v2`
5. Если signal разумный, запускать нужный suite в `full`

Если нужен post-mortem:

1. `main_v2`
2. `pool_ablation_v2`
3. `reward_protocol_ablation_v2`
4. `finetune_ablation_v2`
5. Сравнить markdown tables и comparison figures

## Ограничения текущего v2

В новый pipeline уже встроены:

- shared step context
- reward protocol abstraction
- aligned block-mask training/inference
- script-driven reproducibility
- отдельные artifact namespaces
- обязательные diagnostics

Пока еще не встроены:

- synthetic sanity task
- stronger random-top-k baseline
- 10-seed full rerun как отдельный script preset
- автоматическое обновление `project_report/main.tex`

## Полезные пути

- [pkg](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/pkg)
- [scripts](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/scripts)
- [artifacts_v2](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/artifacts_v2)
- [report_assets_v2](/Users/ayana.mussabayeva/projects/domaha/sequential_desicion_project/report_assets_v2)
