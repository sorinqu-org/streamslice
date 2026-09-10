# Архитектура

## Стадии конвейера

Основной путь — `process_video()` в `src/streamslice/pipeline.py`. Выполняется на уровне часового чанка, затем по каждому выбранному моменту отдельно.

### Уровень чанка (source_path целиком)

| # | Стадия | Модуль | Артефакт |
|---|---|---|---|
| 1 | Определение автора | `identity.resolve_creator_identity` | `creator-profile.json` |
| 2 | Транскрипция (параллельно с 3) | `transcription.transcribe` | `transcript.json` |
| 3 | Анализ аудиопиков (параллельно с 2) | `audio_analysis.analyze_audio` | `audio-peaks.json` |
| 4 | Поиск моментов | `discovery.discover_candidates` | `candidates.json` |
| 5 | Визуальный контроль контекста | `context_analysis.review_candidate_context` | `context-review.json` |
| 6 | Финальный отбор | `curation.curate` | `selection.json` |

Стадии 4–6 работают последовательно над одним и тем же списком кандидатов, отсекая часть на каждом шаге: discovery генерирует до ~10 моментов на окно, context_analysis проверяет самодостаточность и визуальное качество лучших кандидатов, curation отбирает финальные `selection.final_count` штук.

### Уровень клипа (для каждого элемента `selection.json`)

| # | Стадия | Модуль | Артефакт |
|---|---|---|---|
| 7 | Нарезка исходного клипа | `media.cut_source_clip` | `source-clips/clip-XX-source.mp4` (рабочий каталог) |
| 8 | Повторная транскрипция клипа | `transcription.transcribe` + `align_words_to_audio_activity` | `clip-transcript/`, `subtitles.json`, `caption-source.json` |
| 9 | Анализ раскладки кадра | `layout_analysis.analyze_dynamic_layout` | `layout-analysis.json` |
| 10 | Генерация метаданных | `metadata.generate_metadata` | `metadata.json`, `description.txt` |
| 11 | Поиск оверлеев для размытия | `overlay_analysis.detect_overlay_masks` | `overlay-analysis.json` |
| 12 | Подготовка props рендера | `render.prepare_render_props` | `remotion-props.json` |
| 13 | План монтажа + финальные субтитры | `render.plan_montage_and_subtitles` → `montage.plan_edit`, `face_tracking.track_face` | `edit-plan.json`, `face-track.json`, `subtitles.ass` |
| 14 | Рендер | `ffmpeg_render.render_with_ffmpeg` | `clip-XX.mp4` |

Клип пересобирается независимо от чанка: повторная транскрипция (шаг 8) даёт клипу собственную, локальную временную шкалу — субтитры не наследуют таймкоды из транскрипта всего чанка.

Если `render.enabled: false` (режим хоста), стадии 12–14 не выполняются; вместо этого записывается `render-job.json` с указателями на подготовленные `source.mp4` и их SHA-256 — задание для последующего рендера на другой машине (`render_queue.render_job`).

## Кэширование по `*_version`

Каждая дорогая стадия — вызов Gemini или Chromium-рендер — защищена номером версии, зашитым в код:

| Стадия | Константа версии | Файл, где записана |
|---|---|---|
| discovery | `DISCOVERY_VERSION = 8` | `discovery.py` |
| context_analysis | `CONTEXT_REVIEW_VERSION = 6` | `context_analysis.py` |
| curation | `CURATION_VERSION = 7` | `curation.py` |
| identity | `IDENTITY_VERSION = 2` | `identity.py` |
| layout_analysis | `LAYOUT_ANALYSIS_VERSION = 10` | `layout_analysis.py` |
| metadata | `METADATA_VERSION = 5` | `metadata.py` |
| face_tracking | `FACE_TRACK_VERSION = 1` | `face_tracking.py` |
| render | `render.render_version` (конфиг, по умолчанию 11) | `config/default.yaml` |

Логика одинаковая: перед выполнением стадия проверяет, существует ли артефакт на диске и совпадает ли записанное в нём `*_version` с текущей константой в коде. При совпадении результат читается из файла без обращения к Gemini/FFmpeg. При несовпадении (после правки промпта, логики валидации или бизнес-правил) кэш игнорируется и стадия пересчитывается заново.

Рендер клипа (`clip-XX.mp4`) пересчитывается дополнительно при изменении:
- `render.render_version` в конфиге;
- набора масок оверлеев (`overlayMasks`);
- слов субтитров (`words`);
- нормализованной рекомендации раскладки (`layoutRecommendation`).

Сравнение делает `pipeline.process_video`, читая существующий `remotion-props.json` перед тем, как решить, удалять ли готовый `clip-XX.mp4`.

## Структура каталога job/clip

```text
<output.root>/<stream_id>/
├── manifest.json              # сводка запуска: source, model, creator, clips[]
├── transcript.json            # полный транскрипт чанка
├── audio-peaks.json
├── candidates.json            # все найденные кандидаты (до контроля контекста)
├── selection.json             # финальные highlights
├── render-job.json            # только в режиме render.enabled=false
├── DONE.json                  # проставляется после успешного рендера очереди
└── clip-01/
    ├── source.mp4             # точная вырезка исходника для этого клипа
    ├── clip-01.mp4             # финальный вертикальный ролик
    ├── subtitles.json
    ├── subtitles.ass
    ├── metadata.json
    ├── description.txt
    ├── caption-source.json
    ├── overlay-analysis.json
    ├── layout-analysis.json
    ├── remotion-props.json
    ├── edit-plan.json
    ├── face-track.json
    ├── selection.json          # копия highlight-записи для этого клипа
    └── .youtube_uploaded       # маркер после публикации (появляется позже)
```

Рабочий (промежуточный) каталог — `runtime.work_dir` (по умолчанию `.work/<stream_id>/`) — хранит те же JSON на уровне чанка плюс кадры для Gemini Vision (`layout-frames/`, `overlay-frames/`, `context-frames/`) и сырые ответы моделей (`analysis-raw/`, `curator-raw.txt`).

## Контракт `remotion-props.json`

Формируется `render.prepare_render_props`, дополняется `render.plan_montage_and_subtitles` перед рендером.

| Поле | Тип | Источник | Назначение |
|---|---|---|---|
| `source` | str | `prepare_render_props` | Имя файла исходника внутри каталога клипа (`source.mp4`) |
| `title` | str | metadata / clip_reviewer | Заголовок-хук, показывается как ASS-титр |
| `durationInSeconds` | float | `candidate.duration` | Длительность исходного (до монтажа) клипа |
| `renderFps` | int | `render.remotion_fps`/`render.fps` | Целевой FPS |
| `renderVersion` | int | `render.render_version` | Используется для инвалидации кэша рендера |
| `emotionScore` | float | discovery/curation | Оценка эмоциональности 0..10 |
| `layoutRecommendation` | dict | `render.normalized_recommendation` | См. ниже |
| `words` | list[dict] | subtitle_words (клип-локальные) | `{start, end, text}` — источник субтитров и трекинга монтажа |
| `overlayMasks` | list[dict] | `overlay_analysis.detect_overlay_masks` | Боксы для размытия |
| `overlayBlurPx` | int | `overlay_cleanup.blur_px` | Радиус размытия |
| `layout` | dict | `config["layout"]` | Геометрия по умолчанию (webcam/gameplay) — фолбэк, если раскладка не провалидирована |
| `subtitleStyle` | dict | `config["subtitles"]` | Стиль ASS-субтитров |
| `template` | str | `plan_montage_and_subtitles` | Имя разрешённого шаблона (`templates.resolve_template`) |
| `editPlan` | dict | `montage.EditPlan.to_dict()` | План монтажа: список `Cut` + метрики |

`layoutRecommendation` (после нормализации, `render.normalized_recommendation`) может содержать: `layout` (`split`/`webcam_full`/`gameplay_full`), `focus_x`/`focus_y`, `webcam_cut_time`, `webcam_crop`/`webcam_box` (+ `webcam_box_validated`, `webcam_box_confidence`), `event_time`/`event_end`/`gameplay_event_validated`/`gameplay_zoom`/`gameplay_crop_width`, `focal_trajectory`, `time_base` (всегда `"clip"` после нормализации).

## Граница хост / ПК

На хосте (`render.enabled: false`) `process_video` останавливается после подготовки `remotion-props.json` и записи `source.mp4` в каталог клипа; вместо рендера пишется `render-job.json` со списком клипов, их относительных путей и SHA-256 исходников. `render_queue.upload_prepared_job` копирует этот каталог целиком в `queue.remote_jobs` через rclone.

На ПК `render_queue.sync_and_render_queue`:
1. синхронизирует `queue.remote_jobs` → `queue.local_root/incoming`;
2. группирует найденные задания в сессионные батчи по времени (`session_batcher.group_jobs_into_batches`, окно `queue.batch_window_minutes`);
3. если `queue.super_curator_enabled` — сквозной кросс-стримерный отбор топ-клипов внутри батча (`super_curator.select_top_clips`);
4. если `queue.visual_review_enabled` — предрендерный визуальный аудит кадрирования (`clip_reviewer.review_and_refine_clip`);
5. рендерит оставшиеся клипы (`render_queue.render_job` → `render.render_prepared_clip`, который заново строит `plan_montage_and_subtitles` — так изменение шаблона/`montage.*` на ПК подхватывается без переподготовки задания на хосте);
6. выгружает результат в `queue.remote_results`, удаляет исходное задание из очереди, публикует на YouTube (если `youtube.auto_upload`).

Через границу передаются: `source.mp4` для каждого клипа, `remotion-props.json`, `selection.json`, `metadata.json`, `render-job.json`. Часовой исходный чанк на диск ПК никогда не попадает.

## Почему план монтажа и субтитры строятся вместе

`render.plan_montage_and_subtitles` — единственная точка, которая одновременно вызывает `montage.plan_edit` и `subtitles.build_subtitles_ass`. Это не архитектурная случайность, а необходимость: вырезание пауз и ускорение "мёртвого" звука (`montage.*`) физически укорачивает итоговый ролик и сдвигает всё, что идёт после каждого вырезанного куска.

Слова субтитров (`props["words"]`) размечены в исходной, ещё не смонтированной шкале времени клипа. Если бы субтитры строились раньше монтажа (или независимо от него), после вырезания тишины они бы показывались не в момент произнесения реплики, а с постоянно растущим сдвигом — на величину суммарно вырезанного до этой точки времени.

Решение — `EditPlan.remap_words()`: после построения плана (`montage.plan_edit`) слова прогоняются через `EditPlan.map_source_time()`, которая транслирует исходную временную метку в координату уже смонтированного таймлайна. Слово, целиком попавшее в вырезанный кусок, отбрасывается; слово, частично задетое границей выреза, "прилипает" к ближайшей сохранившейся границе (`snap="forward"`/`"back"`). Только после этого пересчитанные слова (`timeline_words`) передаются в `build_subtitles_ass`.

Если план монтажа проходной (`EditPlan.is_passthrough()` — один `Cut` без изменений `zoom`/`speed`, покрывающий весь клип), пересчёт пропускается: субтитры используют исходные `words` без изменений.

## Обработка ошибок и деградация

Разработчики намеренно разделили конвейер на обязательные и необязательные (enrichment) стадии. Обязательные стадии (discovery, context_analysis, curation, транскрипция, рендер) поднимают исключение при полном отказе — `GeminiError`, если не завершилось ни одно окно/кандидат (`discover_candidates`, `review_candidate_context`).

Необязательные стадии по конструкции не могут сорвать рендер клипа:

| Стадия | Поведение при ошибке | Где реализовано |
|---|---|---|
| `layout_analysis.analyze_dynamic_layout` | Логирует предупреждение, возвращает исходную (ненормализованную) рекомендацию | `except GeminiError` |
| `face_tracking.resolve_face_track` | Возвращает `None`; крoп остаётся на геометрии из шаблона/конфига | `except Exception` + `is_usable(min_coverage)` |
| `montage.resolve_edit_plan` | Возвращает `passthrough_plan` — один непрерывный кадр без монтажных эффектов | `except Exception` |
| `overlay_analysis.detect_overlay_masks` при `enabled: false` | Возвращает пустой список масок | явная проверка конфига |
| `clip_reviewer.review_and_refine_clip` (только очередь) | Логирует предупреждение, оставляет `remotion-props.json` без изменений | `except Exception` |
| `identity` model_normalize | Оставляет детерминированный профиль из `config["identity"]["profiles"]` | `except Exception: pass` |

Правило кодовой базы (закреплено в `pyproject.toml`, `per-file-ignores` для `pipeline.py`, `render.py`, `ffmpeg_render.py`): деградационные пути имеют право ловить широкий `Exception`, потому что необрабатываемая ошибка обогащения не должна убивать весь рендер клипа. Каждый такой перехват обязан логировать причину.
