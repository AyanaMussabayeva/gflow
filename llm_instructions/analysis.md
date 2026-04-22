# Analysis aligned with the project task

## Bottom line

С учетом формулировки задания, текущие результаты **не доказывают, что идея GFlowNet-guided Bayesian Optimization несостоятельна**. Они показывают более узкий и честный вывод:

> **В текущей proof-of-concept реализации learned mask generation не демонстрирует устойчивого преимущества над random MC-Dropout mask sampling, а наблюдаемая нестабильность правдоподобно объясняется инженерными и методологическими bottleneck'ами пайплайна.**

Именно так это и стоит подавать в отчете. Это хорошо согласуется с философией задания: ценится не обязательный SOTA-результат, а аккуратный post-mortem, который отделяет отрицательный результат по **текущей реализации** от отрицательного результата по **самой исследовательской гипотезе**.

## Как это соотносится с research question из task

Основной research question в задании сформулирован так: может ли learned mask generation outperform random MC-Dropout for BO exploration, и на каких классах задач это проявляется сильнее.

По текущим экспериментам ответ пока такой:

- **Устойчивого превосходства над random MC-Dropout не показано.**
- **Но и сильного опровержения идеи тоже нет**, потому что картина по benchmark'ам неоднородна и есть признаки того, что главный bottleneck находится в surrogate/reward pipeline.

То есть наиболее корректная формулировка промежуточного результата:

> В рамках текущего PoC мы не наблюдаем надежного выигрыша GFN-генерации масок над random mask sampling. Однако диагностические эксперименты показывают, что этот negative result, вероятно, ограничен качеством текущей реализации proxy reward, surrogate calibration и mask semantics, а потому не должен интерпретироваться как окончательный verdict по самой идее GFN-BO.

Это соответствует духу задания: сделать **rigorous post-mortem** вместо искусственно сильного claim'а.

## Почему сейчас подозрение падает именно на реализацию

### 1. Картина по benchmark'ам слишком неровная для уверенного вывода "идея не работает"

Если бы проблема была именно в исследовательской гипотезе, ожидался бы более стабильный негативный эффект на разных функциях. Вместо этого видно смешанное поведение:

- на части функций GFN уступает random,
- на части функций разница мала относительно разброса по seed'ам,
- на Ackley-10 сигнал уже не выглядит однозначно негативным.

Это больше похоже не на clean falsification идеи, а на **нестабильный экспериментальный pipeline**, чувствительный к детали реализации.

### 2. Даже baseline uncertainty pipeline выглядит нестабильно

В `test.ipynb` уже baseline с MC-Dropout не выглядит по-настоящему надежным. Если даже случайное dropout-based exploration не дает стабильного и предсказуемого поведения, то отрицательный результат для GFN трудно интерпретировать как вывод именно про GFlowNet policy. Сначала нужно убедиться, что сам surrogate uncertainty pipeline вообще работает как разумная основа для BO.

### 3. Есть сильный разрыв между proxy improvement и реальным oracle improvement

Самый тревожный симптом: surrogate/proxy может предсказывать крупное улучшение, а реальная oracle-оценка после запроса оказывается отрицательной. Это уже не просто "GFN не помог". Это свидетельство того, что текущая reward signal для обучения policy **слабо согласована с реальной целью BO**.

А раз policy обучается на такой цели, то отрицательный результат естественно может быть следствием плохой reward construction, а не слабости самой идеи learned mask generation.

### 4. У GFN пока нет убедительных признаков устойчивого обучения

По диагностике обучение policy не выглядит как уверенное смещение распределения в сторону действительно хороших масок. Если TB loss не показывает содержательного выхода на стабильный режим, а sampled rewards лишь слабо отличаются от random baseline, то нельзя утверждать, что мы проверили исследовательскую гипотезу в "сильной" форме. Возможно, GFN просто еще не получает достаточно чистый и информативный training signal.

## Наиболее вероятные bottleneck'и текущей реализации

Ниже — те проблемы, которые стоит прямо описать в отчете как причины, почему negative result пока относится именно к текущему PoC.

### 1. Train/eval mismatch по маскам

Суррогат обучается при одном режиме dropout, а затем оценивается и управляется через другой тип масок (block-wise external masks). Это означает mismatch между тем распределением perturbation'ов, на котором сеть училась, и тем, через которое потом извлекается uncertainty и считается reward.

Почему это критично:

- нарушается интерпретация dropout как approximate posterior sampling;
- качество surrogate under mask становится плохо калиброванным;
- policy может учиться выбирать маски, которые эксплуатируют артефакты masking API, а не meaningful epistemic variation.

### 2. Reward зависит не только от mask quality, но и от случайного candidate pool

Если для каждой маски заново генерируется свой `X_cand`, то reward сравнивает не только маски, но и разные случайные наборы кандидатов. Тогда GFN фактически обучается на шумной смеси двух факторов:

- качества самой маски,
- удачности случайно сэмплированного candidate set.

Это резко повышает дисперсию reward и делает сравнение mask policies методологически нечистым.

### 3. Winner's curse при выборе лучшего кандидата

Внутри каждой маски берется максимум по candidate pool, а затем policy дополнительно концентрируется на масках, которые уже дали экстремальные surrogate scores. Это классическая схема переоптимизма: при шумном surrogate максимумы оказываются смещенными вверх.

Следствие: GFN может переучиваться на маски, которые не находят truly informative regions, а просто чаще производят завышенные surrogate peaks.

### 4. Недостаточная калибровка surrogate

Для нейросетевого BO очень важны:

- нормализация входов,
- стандартизация targets,
- оценка calibration/ranking quality,
- устойчивое сравнение predicted improvement с реальным improvement.

Если этого нет, то провал policy может быть вторичным эффектом: GFN получает плохой reward потому, что surrogate плохо ранжирует кандидатов.

### 5. Нестационарность задачи для GFN почти не контролируется

В самом задании отдельно подчеркивается вопрос non-stationarity: surrogate evolves over time, dataset changes, and GFlowNet should ideally adapt. В текущем PoC это, похоже, еще не проработано как самостоятельная исследовательская ось. Поэтому часть нестабильности может объясняться тем, что policy учится на reward landscape, который быстро дрейфует между BO step'ами.

## Как это лучше подать в отчете

Наиболее сильная и честная позиция — не защищать тезис "у нас просто баг", а формулировать результат как **negative-but-informative PoC**:

> We implemented a proof-of-concept GFN-BO pipeline where a GFlowNet learns to generate dropout masks for a neural BO surrogate. In our current implementation, learned mask generation does not consistently outperform random MC-Dropout mask sampling on Branin, Hartmann-6, and Ackley-10. However, further diagnostics suggest that the dominant bottlenecks lie in reward construction, mask-train/eval mismatch, and surrogate calibration. Therefore, our findings should be interpreted as a post-mortem of the current implementation rather than a definitive rejection of GFN-guided exploration.

Это формулировка, которая:

- соответствует критериям курса;
- не overstating claims;
- показывает исследовательскую зрелость;
- естественно ведет к разделу limitations / future work.

## Конкретный план по улучшению кодовой базы

Ниже — план, который одновременно усилит и саму систему, и качество итогового отчета.

### Priority 1. Сделать mask pipeline согласованным и тестируемым

Вынести всю mask logic в отдельный модуль:

- `src/masks.py`
- `BlockMask`
- `sample_mask`
- `apply_mask`
- `split_mask_bits`
- единая семантика scaling/keep probability

Что изменить:

- обучать surrogate под тем же типом masks, который затем используется для inference/policy;
- убрать неявные различия между training dropout и external masking;
- добавить unit tests на shape, scaling, determinism и equivalence semantics.

Почему это важно для отчета:

- можно честно написать, что первоначальный negative result был partially confounded by a mask mismatch;
- после фикса появится clean ablation "before vs after mask alignment".

### Priority 2. Фиксировать candidate pool внутри BO step

Для каждого шага BO нужно заранее генерировать один и тот же `X_pool` и использовать его:

- для всех masks,
- для random baseline,
- для GFN policy,
- для surrogate diagnostics.

Лучше использовать Sobol / Latin Hypercube вместо независимого iid uniform sampling.

Почему это важно:

- reward становится существенно менее шумным;
- сравнение masks становится fair;
- появляется воспроизводимость эксперимента на уровне одного BO step.

### Priority 3. Разделить proposal и evaluation

Сейчас одна и та же модель может и предлагать, и подтверждать кандидата, что усиливает winner's curse.

Лучше сделать двухстадийную схему:

1. mask proposes top candidates;
2. held-out committee of masks / held-out dropout ensemble reranks them;
3. только после этого выбирается `x_next`.

Альтернативы:

- `top-k average` вместо single best;
- lower-confidence reranking;
- CVaR-style conservative score.

Это сделает claims про improvement намного более правдоподобными.

### Priority 4. Добавить полноценную surrogate calibration диагностику

Нужно логировать не только regret curves, но и surrogate quality metrics:

- RMSE / MAE на holdout candidate pool,
- rank correlation between surrogate score and oracle value,
- top-k hit rate,
- calibration gap for predicted improvements.

Также обязательно:

- нормализовать `X`,
- стандартизовать `y`,
- фиксировать random seeds и train splits.

Это крайне полезно для отчета: если GFN проигрывает, но surrogate ranking quality тоже плохая, то causal story становится гораздо убедительнее.

### Priority 5. Переформулировать proxy reward как отдельную исследовательскую ось

Сейчас proxy reward — вероятно, главный источник шума. Поэтому лучше сделать его объектом controlled ablation.

Минимум три варианта reward:

- raw predicted improvement,
- rank-based reward,
- held-out ensemble reward.

Дополнительно полезно попробовать:

- clipped reward,
- z-score normalization across masks,
- moving-baseline reward relative to current candidate set.

Тогда в отчете можно будет ответить на более содержательный вопрос:

> The main failure mode was not GFlowNet training per se, but the mismatch between the proxy reward and actual oracle improvement.

### Priority 6. Сделать sanity checks, отделяющие идею от реализации

Это, пожалуй, самый важный шаг для research-quality narrative.

Нужны хотя бы три sanity experiments:

1. **Synthetic mask-reward task**  
   На искусственной задаче, где истинное распределение по маскам контролируется, GFN должен обучаться лучше uniform random. Это докажет, что policy implementation вообще работает.

2. **Oracle-assisted reward check**  
   На маленькой задаче можно считать более точный reward, приближенный к oracle. Если там GFN начинает выигрывать, значит failure source — proxy approximation.

3. **Random-top-k baseline**  
   Сравнить GFN не только с random mask sampling, но и с stronger baseline: many random masks + best-of-N selection. Это покажет, добавляет ли GFN structure beyond brute force search.

Именно такие sanity checks особенно хорошо соответствуют философии задания про PoC и controlled environments.

### Priority 7. Усилить статистическую строгость

Текущих 5 seed'ов мало для уверенных выводов. Лучше перейти к:

- 20 seeds,
- paired experiment design,
- одинаковым initial datasets,
- одинаковым candidate pools,
- bootstrap confidence intervals,
- paired test on final regret and regret AUC.

Тогда даже отрицательный результат будет выглядеть как добротный workshop-style empirical finding, а не как anecdotal outcome.

### Priority 8. Перестроить репозиторий под reproducible codebase

Поскольку deliverable включает reproducible repository, стоит уйти от логики в ноутбуках и собрать код так:

- `src/benchmarks.py`
- `src/surrogate.py`
- `src/masks.py`
- `src/proxy_reward.py`
- `src/gfn_policy.py`
- `src/bo_loop.py`
- `src/eval.py`
- `configs/*.yaml`
- `scripts/run_benchmark.py`
- `tests/*.py`

Минимальный набор тестов:

- `test_mask_semantics.py`
- `test_reward_determinism.py`
- `test_seed_reproducibility.py`
- `test_gfn_synthetic_learning.py`

Это усилит не только код, но и сам отчет: можно будет утверждать, что conclusions were reproduced across a shared, tested implementation.

## Что делать в первую очередь, если времени мало

Если нужен максимально практичный short list, я бы делал в таком порядке:

1. **Fix train/eval mask mismatch**  
2. **Use a fixed candidate pool per BO step**  
3. **Add normalization + surrogate diagnostics**  
4. **Run paired experiments with more seeds**  
5. **Add one sanity experiment with a cleaner reward**

Эти пять шагов дадут наибольший прирост к качеству выводов при умеренных усилиях.

## Финальная формулировка для report/discussion section

Можно использовать почти дословно:

> Our current experiments do not provide evidence that GFlowNet-guided mask generation consistently improves Bayesian optimization over random MC-Dropout mask sampling. However, we found several strong confounders in the current proof-of-concept implementation: mismatch between training-time and evaluation-time masking, high-variance proxy rewards, limited surrogate calibration, and insufficient control over non-stationarity. We therefore interpret our findings as a negative result about the present implementation, not a definitive rejection of the broader GFN-BO hypothesis. In the context of the course project, this constitutes a meaningful post-mortem that identifies concrete empirical bottlenecks and motivates a cleaner next iteration.

## Final recommendation

Да, **в отчете стоит заметно сместить акцент с "получилось/не получилось beat baseline" на "что именно было проверено, почему это пока не дало стабильного выигрыша и какие bottleneck'и были выявлены"**. Для этого задания это не слабая позиция, а наоборот правильная исследовательская позиция.

Самая сильная версия narrative здесь такая:

- мы реализовали осмысленный PoC для GFN-guided BO;
- устойчивого выигрыша пока не увидели;
- но смогли локализовать наиболее вероятные причины неуспеха;
- предложили конкретную программу улучшений, которая делает следующий iteration scientifically cleaner.

Именно такая подача лучше всего соответствует формулировке `task.txt` и критериям оценки проекта.
