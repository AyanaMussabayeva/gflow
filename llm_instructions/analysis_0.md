План усиления кодовой базы

Починить reward protocol в gfn_bo_experiments.py (line 434).
На каждой BO-итерации один раз сэмплировать общий candidate_pool и общий набор heldout_masks, а потом использовать их для всех mask evaluation. Это сразу уберет главный источник шума и сделает сравнение mask-ов честным.

Переписать reward scaling.
Вместо текущего softplus(improvement / tau) ввести нормализацию по батчу или rank-based reward. Цель: чтобы на Branin reward не схлопывался в floor. Минимальный smoke-test: reward variance на Branin должна быть строго > 0.

Согласовать training/inference mask distribution.
Либо обучать surrogate теми же block-wise masks, которые потом ищет GFN, либо уменьшить inference keep/drop pattern так, чтобы он соответствовал dropout_p. Сейчас это слишком разные режимы.

Добавить диагностику в результаты.
В run_single_trial и run_compute_tradeoff_experiment.py логировать:
reward std, improvement std, долю floor-reward, корреляцию proxy_improvement -> actual_gain, повторную оценку одной mask.
Это даст очень сильный appendix и защитит отчет.

Добавить 3 обязательные абляции.
shared candidate pool vs fresh candidate pool, raw improvement vs normalized/rank reward, continual finetune vs retrain from scratch.
Именно они отделят “идея слабая” от “реализация зашумлена”.

После фиксов перезапустить только ключевой набор.
Branin, Hartmann6, Ackley10, но уже с 10 seed-ами. Не нужен большой benchmark zoo; важнее показать, что после исправления протокола выводы стабильны.

Переписать выводы в отчете под post-mortem style.
Если после фикса Branin/Hartmann все еще слабые, это уже будет сильный научный negative result. Если улучшатся, отчет станет заметно убедительнее, потому что ты покажешь, что отделил implementation artifact от method effect.
