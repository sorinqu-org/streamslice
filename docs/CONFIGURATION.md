# Конфигурация

Конфиг — YAML с наследованием через `extends` (`config.py`, `_deep_merge`): дочерний файл переопределяет только указанные ключи, остальное наследуется от родителя рекурсивно для вложенных словарей. `config/remote.yaml`, `config/local-render.yaml` и `config/test.yaml` все указывают `extends: default.yaml`.

Секреты (API-ключи, токены) в YAML не хранятся — см. [«Переменные окружения»](#переменные-окружения).

## `paths` / `project_path`

Отдельной секции `paths` в конфиге нет. Вместо неё большинство путей в `config/default.yaml` — абсолютные (`/home/yuwye/...`, `/opt/...`). Часть путей — относительные к репозиторию (`sync.log_file: .work/rclone.log`, `render.remotion_dir: remotion`) и резолвятся через `config.project_path(config, value)`:

```python
def project_path(config, value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config["_project_root"]) / path
    return path.resolve()
```

`_project_root` — родитель каталога, где лежит YAML-файл конфига (`config_path.parent.parent`), то есть корень репозитория при обычной раскладке `config/*.yaml`.

## Обязательные секции и валидация

`config._validate()` требует наличия следующих секций верхнего уровня — при отсутствии любой из них `load_config` бросает `ConfigError: Missing config sections: ...`:

```
proxy, models, sync, transcription, audio_analysis, selection, layout, subtitles, render, output, runtime
```

Дополнительные проверки значений:

| Проверка | Условие ошибки | Сообщение |
|---|---|---|
| `selection.min_duration_seconds` | `< 20` | `selection.min_duration_seconds must be at least 20` |
| `selection.max_duration_seconds` | `> 90` | `selection.max_duration_seconds must be at most 90` |
| `selection.final_count` | вне `[1, 10]` | `selection.final_count must be between 1 and 10` |
| `render.width, render.height, render.fps` | `!= (1080, 1920, 60)` | `render must be 1080x1920 at 60 FPS` |

`face_tracking`, `montage`, `templates` и другие секции, читаемые через `config.get(...)`, не обязательны — их отсутствие тихо заменяется дефолтами в коде.

## Переменные окружения

| Переменная | Где используется | Назначение |
|---|---|---|
| `CLIPROXY_API_KEY` | `proxy.api_key_env` (по умолчанию) | Bearer-токен для CLIProxyAPI. Обязателен для любого вызова Gemini |
| `/etc/streamslice.env` | `tools/process_chunk_for_queue.py` | На хосте подхватывается вручную (`_load_env_file`) для запуска не через systemd |

Приоритет: `proxy.api_key_env` в YAML задаёт **имя** переменной окружения, само значение всегда берётся из окружения процесса — в YAML ключ никогда не хранится. Если переменная пуста, `proxy.api_key()` бросает `ProxyError: Set CLIPROXY_API_KEY before running`.

## Три готовых конфига

| Файл | Роль | Отличия от `default.yaml` |
|---|---|---|
| `config/default.yaml` | база, разработка на своей машине | Полный набор секций, `render.enabled: true`, `sync.enabled: true`, `queue.enabled: false` |
| `config/remote.yaml` | хост (VPS) | `render.enabled: false`, `chromium_executable: /bin/false`, `sync.enabled: false` (чанки приходят от recorder напрямую), `queue.enabled: true` с путями `/var/lib/streamslice/...`, `runtime.ffmpeg_preset: ultrafast`/`crf: 23` (черновая нарезка source-клипов — не финальное качество) |
| `config/local-render.yaml` | локальный ПК с GPU | `sync.enabled: false`, `queue.enabled: true` с путями `~/tiktoks/render-queue`, `youtube.headless: true` + `cookies_file` (Playwright-логин, устаревший путь) |
| `config/test.yaml` | быстрый прогон для проверки | `render.scale: 0.5`, `render.remotion_fps: 30` — уменьшенное разрешение чернового Remotion-рендера (актуально только для `render.engine: remotion`, у FFmpeg-движка эти поля не используются) |

## Секции

### `proxy`

| Ключ | Тип | По умолчанию | Что делает | Когда менять |
|---|---|---|---|---|
| `base_url` | str | `http://127.0.0.1:8081/v1` | Базовый URL CLIProxyAPI (OpenAI-совместимый) | Меняете порт/адрес прокси |
| `api_key_env` | str | `CLIPROXY_API_KEY` | Имя переменной окружения с ключом | Почти никогда |
| `binary` | str | путь к `cli-proxy-api` | Бинарник, который запускается автоматически, если прокси не отвечает (`proxy.start_proxy`) | Меняете расположение CLIProxyAPI |
| `config` | str | путь к `config.yaml` CLIProxyAPI | Конфиг самого прокси (не StreamSlice) | — |
| `working_dir` | str | каталог CLIProxyAPI | Рабочая директория процесса прокси | — |
| `log_file` | str | путь к `proxy.log` | Куда пишется stdout/stderr прокси | — |
| `startup_timeout_seconds` | int | `10` | Сколько ждать прокси после автозапуска (`ensure_proxy`) | Медленный старт CLIProxyAPI |

### `providers`

Именованные HTTP-провайдеры для секции `models` (см. `gemini._provider_settings`). Сейчас единственный активно используемый — `cliproxy` (алиас на `proxy.*`). Секция `openrouter`, присутствовавшая в старых версиях `default.yaml`, удалена — OpenRouter в проекте не используется, все модели маршрутизируются через CLIProxyAPI.

### `models`

Плоское отображение "роль → имя модели" либо `{provider: <имя>, model: <имя>}` для нестандартного провайдера (см. `gemini.GeminiClient._resolve_target`).

| Ключ | По умолчанию | Что делает |
|---|---|---|
| `transcription` | `gemini-3.8-flash-high` | Word-level транскрипция аудио |
| `analysis` | `gemini-3.8-flash-high` | Поиск кандидатов в моменты (`discovery.py`) |
| `curator` | `gemini-3.8-flash-high` | Финальный отбор (`curation.py`), контроль контекста (`context_analysis.py`) |
| `metadata` | `gemini-3.8-flash-high` | Заголовок/описание/хэштеги, нормализация профиля автора |
| `overlay_detection` | `gemini-3.8-flash-high` | Поиск оверлеев для размытия (`overlay_analysis.py`), анализ раскладки кадра (`layout_analysis.py`) |
| `visual_review` | *(не задан)* | Опциональная роль для `clip_reviewer.py`; при отсутствии используется `overlay_detection`, затем `curator` |

### `identity`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `model_normalize` | bool | `true` | Разрешает Gemini лишь причёсывать формулировку имени/алиасов автора, не выдумывая новые факты |
| `profiles.<key>` | dict | — | Профиль конкретного Twitch-логина: `twitch_login`, `display_name`, `aliases`, `preferred_mentions`, `real_name`, `source_url`, `character_lore` (используется в промптах discovery/curation/metadata как контекст о стримере) |

`<key>` профиля — это верхний уровень пути к записям стрима внутри `sync.local_dir` (обычно совпадает с Twitch-логином).

### `sync`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `enabled` | bool | `true` | Включает команды `sync`/`run`/`run-latest` |
| `remote` | str | `gdrive:Twitch` | rclone remote с записями стримов |
| `local_dir` | str | `/home/yuwye/streams` | Куда скачиваются файлы |
| `clean_before_copy` | bool | `true` | Требует `--confirm-cleanup`; удаляет старые видео/кэш в `local_dir` перед копированием |
| `video_extensions` | list[str] | `[.mp4, .mkv, .mov, .webm]` | Какие файлы считаются видео при очистке |
| `cache_names` | list[str] | `[.cache, .pipeline-cache]` | Какие каталоги считаются кэшем при очистке |
| `transfers` / `checkers` | int | `2` / `4` | Параллелизм rclone |
| `multi_thread_streams` | int | `8` | rclone multi-thread копирование одного файла |
| `multi_thread_chunk_size` | str | `16Mi` | Размер чанка многопоточной загрузки |
| `log_file` | str | `.work/rclone.log` | Лог rclone (резолвится через `project_path`) |

### `transcription`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `language` | str | `ru` | Язык транскрипции |
| `segment_seconds` | int | `240` | Длина одного сегмента аудио, отправляемого в Gemini |
| `overlap_seconds` | int | `2` | Перехлёст между сегментами (склейка слов на границе) |
| `audio_bitrate` | str | `48k` | Битрейт извлекаемого MP3 |
| `sample_rate` | int | `16000` | Частота дискретизации |
| `parallel_requests` | int | `2` | Параллельные запросы транскрипции |
| `request_timeout_seconds` | int | `300` | Таймаут одного запроса |
| `minimum_coverage_ratio` | float | `0.35` | Минимальная доля покрытия сегмента распознанным текстом, ниже которой сегмент считается подозрительным |

### `audio_analysis`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `sample_rate` | int | `16000` | Частота при извлечении PCM для анализа громкости |
| `window_seconds` | float | `0.5` | Окно RMS |
| `peak_z_score` | float | `1.35` | Порог z-score для признания окна пиком |
| `merge_distance_seconds` | float | `4` | Слияние близких пиков |
| `max_peaks` | int | `100` | Максимум пиков в результате |
| `use_librosa_if_available` | bool | `true` | Использовать librosa (если установлен extras `audio`) для более точного onset-детектора |

### `selection`

Управляет поиском (discovery), контролем контекста и финальным отбором (curation).

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `analysis_window_seconds` | int | `600` | Размер окна транскрипта, отправляемого в discovery за раз |
| `analysis_overlap_seconds` | int | `60` | Перехлёст окон discovery |
| `parallel_requests` | int | `4` | Параллельные окна discovery |
| `min_duration_seconds` | int | `20` | Мин. длина кандидата (валидируется `config.py`, нельзя поставить < 20) |
| `max_duration_seconds` | int | `90` | Макс. длина кандидата (валидируется `config.py`, нельзя поставить > 90) |
| `pre_context_min_seconds` / `pre_context_max_seconds` | int | `5` / `15` | Разброс завязки перед событием (используется как ориентир в промпте discovery) |
| `post_context_min_seconds` / `post_context_max_seconds` | int | `5` / `10` | Разброс развязки после события |
| `final_count` | int | `2` | Сколько клипов оставить после curation (валидируется, 1..10) |
| `max_candidates_for_curator` | int | `60` | Сколько лучших (по эвристическому скору) кандидатов реально отправляется в curator-промпт |
| `strict_context` | bool | `true` | Включает `context_analysis.review_candidate_context`; при `false` стадия отключена и кандидаты идут в curation без визуального контроля |
| `min_context_score` | float | `5.5` | Порог `context_score` для допуска кандидата |
| `min_quality_score` | float | `6.0` | Порог `quality_score` |
| `context_review_limit` | int | `30` | Сколько кандидатов реально прогоняется через визуальный контроль (топ по эвристике + топ по эмоции) |
| `context_parallel_requests` | int | `3` | Параллелизм визуального контроля |
| `allow_fewer_if_context_weak` | bool | `false` | *(читается конфигом, не используется напрямую в найденном коде отбора — влияет на общее поведение "не заполнять квоту")* |

`context_analysis.py` дополнительно жёстко требует (не через конфиг) `hook_score >= 6.5`, `payoff_score >= 6.5`, `shareability_score >= 6.5` и минимум два элемента `evidence` — эти пороги не вынесены в YAML.

### `layout`

Геометрия по умолчанию для FFmpeg-рендера, если Gemini/шаблон её не переопределили.

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `webcam.x/y/width/height` | float | `0.738/0.739/0.262/0.261` | Фолбэк-координаты вебки в исходном кадре (0..1) |
| `webcam_full.x/y/width/height` | float | `0.780/0.740/0.220/0.260` | Фолбэк для полноэкранного лица |
| `gameplay.x/y/width/height` | float | `0.0/0.0/0.738/1.0` | Фолбэк для области геймплея |
| `split_webcam_height` / `split_gameplay_height` | float | `0.38` / `0.62` | *(унаследованы из дорендерной эпохи; актуальная высота полос задаётся в шаблоне, см. [TEMPLATES.md](TEMPLATES.md))* |
| `fullscreen_emotion_threshold` | float | `8.5` | Используется как ориентир для промптов, определяющих переход в webcam_full |
| `webcam_detection_min_confidence` | float | `0.75` | Порог `webcam_box_confidence`, ниже которого визуально найденный кроп вебки отбрасывается (`layout_analysis._validated_webcam`) |
| `zoom_scale` | float | `1.28` | Референсное значение зума для промптов |
| `default_focus_x` / `default_focus_y` | float | `0.50` | Фокус по умолчанию, если ни Gemini, ни трекинг лица ничего не дали |

### `overlay_cleanup`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `enabled` | bool | `true` | Включает поиск и размытие рекламных/донатных оверлеев |
| `sample_positions` | list[float] | `[0.12, 0.50, 0.88]` | Относительные тайминги кадров, отправляемых в Gemini Vision |
| `min_confidence` | float | `0.65` | Порог `confidence` для принятия найденного бокса |
| `max_boxes` | int | `8` | Максимум масок на клип |
| `padding` | float | `0.012` | Расширение бокса (0..1 от кадра) на всякий случай |
| `blur_px` | int | `28` | Радиус размытия (`overlayBlurPx` в `remotion-props.json`) |

### `layout_analysis`

Отдельной секции `layout_analysis` в `config/default.yaml` нет: `analyze_dynamic_layout` (модуль `layout_analysis.py`) читает пороги из `layout.webcam_detection_min_confidence` и версионируется константой `LAYOUT_ANALYSIS_VERSION` в коде, а не конфигом.

### `context_analysis`

Отдельной секции нет — параметры контроля контекста (`min_context_score`, `min_quality_score`, `context_review_limit`, `context_parallel_requests`, `strict_context`) находятся внутри `selection` (см. выше).

### `clip_reviewer`

Опциональная секция для предрендерного визуального аудита в очереди (`clip_reviewer.review_and_refine_clip`), в `default.yaml` не объявлена — код использует дефолт.

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `sample_positions` | list[float] | `[0.10, 0.30, 0.50, 0.80]` | Тайминги кадров для визуального аудита кадрирования/заголовка перед финальным рендером в очереди |

Включение/выключение самой стадии — через `queue.visual_review_enabled`.

### `transcription` *(клип-локальная повторная транскрипция)*

Использует ту же секцию `transcription`, что и транскрипция чанка — отдельных ключей для клипа нет.

### `subtitles`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `font_family` | str | `Montserrat` | Шрифт ASS-стиля `TikTok` |
| `font_size` | int | `92` | Размер шрифта |
| `primary_color` | str (hex) | `#FFE815` | Основной цвет текста (в ASS переводится в `&H0015E8FF`, порядок BGR) |
| `active_color` | str (hex) | `#FFE815` | Цвет активного слова *(поле присутствует в конфиге и в Remotion-типах; текущий ASS-рендер `subtitles._to_ass` использует единый стиль без отдельной подсветки активного слова)* |
| `outline_color` | str (hex) | `#000000` | Цвет обводки |
| `outline_width` | int | `14` | Толщина обводки |
| `shadow_blur` | int | `18` | *(поле конфигурации; в ASS-генераторе `_to_ass` не задействовано напрямую — тень задаётся стилем `BorderStyle`/`Shadow` в шаблоне ASS)* |
| `words_per_group` | int | `1` | Сколько слов показывать одной репликой субтитров |
| `mask_profanity` | bool | `true` | Маскировать мат через `profanity.mask_profanity` |

### `render`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `enabled` | bool | `true` | `false` на хосте — рендер откладывается, готовится только `render-job.json` |
| `engine` | str | `ffmpeg` | `ffmpeg` (актуальный нативный рендер) или `remotion` (устаревший Chromium-путь через `_render_with_props`) |
| `width` / `height` / `fps` | int | `1080` / `1920` / `60` | Жёстко валидируется `config.py`, менять нельзя |
| `codec` | str | `h264` | Фолбэк-кодек, если `upscale_encoder` не задан |
| `crf` | int | `18` | CRF для CPU-пути (`libx264`), если аппаратное ускорение недоступно/отключено |
| `scale` | float | `1.0` | Множитель разрешения при `engine: remotion` (Remotion рендерит в меньшем разрешении, затем FFmpeg апскейлит) |
| `remotion_fps` | int | `60` | FPS, в котором Remotion считает кадры до апскейла (актуально только для `engine: remotion`) |
| `upscale_encoder` | str | `h264_nvenc` | Кодек GPU-энкодера. `h264`/`libx264` автоматически заменяются на `h264_nvenc` в `render_with_ffmpeg` |
| `concurrency` | int | `4` | Параллелизм Chromium (`engine: remotion`) |
| `chromium_executable` | str | `/usr/bin/google-chrome-stable` | Бинарник Chrome/Chromium (только `engine: remotion`; на хосте ставится `/bin/false`, потому что рендер там выключен) |
| `hardware_acceleration` | str | `required` | Режим GPU для Remotion (`engine: remotion`) |
| `offthread_video_threads` | int | `4` | Потоки декодирования Remotion |
| `gl_renderer` | str | `angle-egl` | GL-бэкенд Chromium |
| `video_bitrate` | str | `8M` | Целевой битрейт NVENC (используется и FFmpeg-, и Remotion-путём) |
| `cq` | int | `19` | Constant quality для NVENC (`ffmpeg_render.render_with_ffmpeg`, дефолт в коде) |
| `render_version` | int | `11` | Инвалидация кэша готового `clip-XX.mp4` при изменении логики рендера |
| `remotion_dir` | str | `remotion` | Каталог Remotion-проекта (используется только `engine: remotion`) |
| `timeout_seconds` | int | `1800` | Таймаут одного рендера |
| `template` | str | `classic-split` | Имя шаблона по умолчанию (см. [TEMPLATES.md](TEMPLATES.md)) |
| `templates_dir` | str | *(не задан)* | Дополнительный каталог с пользовательскими шаблонами, высший приоритет поиска |
| `ffmpeg_preset` | str | *(нет в `render`, см. `runtime.ffmpeg_preset`)* | — |

### `face_tracking`

Секции нет в `config/default.yaml` — все ключи опциональны, код (`face_tracking.py`, `ffmpeg_render.resolve_face_track`) использует дефолты из кода при отсутствии.

| Ключ | Тип | Дефолт в коде | Диапазон | Что делает |
|---|---|---|---|---|
| `enabled` | bool | `true` | — | Включает трекинг; `false` — крoп остаётся на геометрии из шаблона/конфига |
| `sample_fps` | float | `4.0` | 2–8 | Частота выборки кадров для детектора YuNet. Выше — точнее и медленнее |
| `min_coverage` | float | `0.4` | 0–1 | Минимальная доля кадров с найденным лицом; ниже — трек считается неудачным и отбрасывается (`FaceTrack.is_usable`) |
| `model_path` | str | путь к `assets/models/face_detection_yunet_2023mar.onnx` | — | Путь к ONNX-модели YuNet |
| `detect_width` | int | `640` | 320–960 | Ширина кадра, подаваемого в детектор (после кропа под регион поиска) |
| `min_detect_width` | int | `320` | — | Если декодированная ширина региона меньше — регион апскейлится перед детекцией |
| `score_threshold` | float | `0.6` | 0–1 | Порог уверенности YuNet |
| `nms_threshold` | float | `0.3` | 0–1 | Порог non-max suppression |
| `max_gap_seconds` | float | `1.0` | — | Максимальный разрыв между сэмплами, который ещё интерполируется; больше — трек считается прерванным |
| `median_window` | int | `5` | 3–9, нечётное | Окно временного медианного фильтра (первый этап сглаживания) |
| `deadband` | float | `0.015` | 0.005–0.03 | Мёртвая зона (в долях кадра): движение меньше порога игнорируется |
| `ema_alpha` | float | `0.25` | 0.1–0.5 | Коэффициент экспоненциального сглаживания (выше — быстрее реагирует, но дёрганее) |
| `max_velocity` | float | `0.35` | 0.15–0.6 | Максимальная скорость смещения центра кропа, в долях кадра/сек |
| `upscale` | float | `2.0` | 1–3 | Множитель апскейла региона перед детекцией, если он меньше `min_detect_width` |

Подробное объяснение конвейера сглаживания — в [MONTAGE.md](MONTAGE.md#трекинг-лица).

### `montage`

Секции нет в `config/default.yaml` — дефолты определены в `montage._DEFAULTS`. Полное описание — в [MONTAGE.md](MONTAGE.md).

| Ключ | Тип | Дефолт в коде | Диапазон | Что делает |
|---|---|---|---|---|
| `enabled` | bool | `true` | — | Выключает весь монтажный план (`passthrough_plan`) |
| `trim_silence` | bool | `true` | — | Вырезать длинные паузы |
| `max_silence_seconds` | float | `1.2` | 0.6–3.0 | Пауза короче этого порога не вырезается |
| `silence_padding_seconds` | float | `0.25` | 0.1–0.5 | Отступ от границ паузы, который остаётся нетронутым |
| `protect_head_seconds` | float | `2.0` | 1–4 | Начало клипа (хук), которое никогда не режется |
| `protect_tail_seconds` | float | `1.5` | 1–3 | Конец клипа (развязка), который никогда не режется |
| `max_removed_ratio` | float | `0.25` | 0.1–0.4 | Максимальная доля длительности клипа, которую можно вырезать суммарно |
| `min_segment_seconds` | float | `1.5` | 0.5–3 | Сегменты раскладки короче этого сливаются с соседними |
| `event_layout` | str | `split` | `split`/`webcam_full`/`gameplay_full` | Раскладка на время подтверждённого визуального события |
| `punch_in` | bool | `true` | — | Включает punch-in зум на аудиопиках |
| `punch_zoom` | float | `1.10` | 1.05–1.25 | Множитель зума на punch-in |
| `punch_seconds` | float | `1.2` | 0.6–2.0 | Длительность одного punch-in |
| `punch_cooldown_seconds` | float | `6.0` | 3–10 | Минимальный интервал между punch-in |
| `max_punches` | int | `4` | 1–6 | Максимум punch-in на клип |
| `hook_punch` | bool | `true` | — | Лёгкий зум в первые секунды клипа |
| `hook_seconds` | float | `1.5` | 1.0–2.5 | Длительность хук-зума |
| `hook_zoom` | float | `1.06` | 1.02–1.10 | Множитель хук-зума |
| `speed_up_silence` | bool | `false` | — | Вместо вырезания — ускорение пауз (альтернатива `trim_silence`) |
| `silence_speed` | float | `1.6` | 1.2–2.5 | Множитель скорости при `speed_up_silence: true` |

### `metadata`

Отдельной секции нет — генерация заголовков/описаний использует `models.metadata` и версионируется `METADATA_VERSION` в коде (`metadata.py`).

### `youtube`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `auto_upload` | bool | `true` | Автопубликация клипов после рендера в очереди (`render_queue._materialize_output`) |
| `interval_minutes` | int | `12` | Пауза между публикациями клипов подряд (защита от shadowban) |
| `client_secrets_file` | str | `~/.config/streamslice/client_secrets.json` | *(путь для справки; фактически `youtube_api_uploader.py` ищет секреты в `~/.config/streamslice/client_secrets.json` или `~/.config/streamslice/secrets/client_secrets_*.json` для ротации нескольких проектов — см. `get_secrets_pool`)* |
| `token_file` | str | `~/.config/streamslice/youtube_token.json` | Аналогично — реальный путь берётся из `DEFAULT_TOKEN_PATH`, ключ носит справочный характер |
| `visibility` | str | `public` | `privacyStatus` в теле запроса YouTube Data API |
| `not_made_for_kids` / `contains_ai` / `paid_promotion` / `standard_license` / `allow_embedding` / `publish_to_feed` / `auto_remove_copyright_music` | bool | `true`/`false`/… | Заданы в `default.yaml`, но актуальный загрузчик (`youtube_api_uploader.upload_shorts_api`, YouTube Data API v3) их не читает — жёстко проставляет `selfDeclaredMadeForKids: False`, `embeddable: True`, `publicStatsViewable: True`. Эти ключи относятся к устаревшему Playwright-загрузчику (`youtube_uploader.py`), который тоже не читает их как конфиг — значения там захардкожены в селекторах UI Studio |
| `max_age_days` | float | `2.0` | Используется только `retry_youtube_uploads.py` (CLI `--max-age-days`): пропускает клипы старше указанного числа дней при повторной публикации |
| `cookies_file` / `headless` | str / bool | — | Только для устаревшего Playwright-пути (`config/local-render.yaml`) |

### `gemini`

Отдельной секции `gemini` в конфиге нет — все настройки доступа к модели находятся в `proxy` и `providers`, см. выше.

### `proxy`

См. раздел [`proxy`](#proxy) выше.

### `identity`

См. раздел [`identity`](#identity) выше.

### `profanity`

Секции конфига нет. `profanity.py` — набор жёстко заданных regex-паттернов для русского мата, без настраиваемых параметров. Включение маскировки — через `subtitles.mask_profanity`.

### `audio`

Секции с именем `audio` нет — есть `audio_analysis` (анализ пиков громкости для discovery, см. выше) и extras-зависимость `audio` в `pyproject.toml` (`librosa`), включаемая флагом `audio_analysis.use_librosa_if_available`.

### `sync`

См. раздел [`sync`](#sync) выше.

### `queue`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `enabled` | bool | `false` | Требуется `true` для команды `render-queue` |
| `remote_jobs` | str | `gdrive:StreamSlice/render-queue` | rclone remote с входящими заданиями |
| `remote_results` | str | `gdrive:StreamSlice/render-results` | rclone remote для результатов |
| `local_root` | str | `/home/yuwye/tiktoks/render-queue` | Локальный рабочий каталог очереди (`incoming/`) |
| `rclone_config` | str | путь к `rclone.conf` | Явный путь к конфигу rclone, если не системный |
| `transfers` / `checkers` | int | `4` / `8` | Параллелизм rclone для очереди |
| `drive_chunk_size` | str | `64Mi` | Размер чанка загрузки на Google Drive |
| `max_parallel_renders` | int | `3` (`1` в `config/remote.yaml`, где рендер всё равно отключён) | Сколько клипов одного задания рендерятся одновременно |
| `max_parallel_jobs` | int | `1` | Сколько заданий обрабатывается одновременно |
| `sync_timeout_seconds` | int | `7200` (`14400` на VPS) | Таймаут rclone-операций синхронизации очереди |
| `batch_window_minutes` | int | `30` | Окно группировки заданий в сессионные батчи (`session_batcher.group_jobs_into_batches`) |
| `super_curator_enabled` | bool | `true` | Кросс-стримерный отбор топ-клипов внутри батча |
| `max_batch_clips` / `min_batch_clips` | int | `3` / `2` | Границы отбора `super_curator.select_top_clips` |
| `visual_review_enabled` | bool | `true` | Предрендерный визуальный аудит (`clip_reviewer.review_and_refine_clip`) |

## Ключи `runtime`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `work_dir` | str | `.work` | Рабочий каталог промежуточных файлов (резолвится через `project_path`) |
| `retries` | int | `8` | Число повторов запроса к Gemini при сбое |
| `retry_initial_seconds` / `retry_max_seconds` | int | `2` / `30` | Экспоненциальная задержка между повторами |
| `ffmpeg_preset` | str | `veryfast` (`ultrafast` на хосте) | Пресет `libx264` при нарезке исходных клипов и CPU-фолбэке рендера |
| `ffmpeg_crf` | int | `18` (`23` на хосте) | CRF при нарезке исходных клипов |
| `duration_tolerance_seconds` | float | `0.35` | Допустимое расхождение фактической длительности вырезанного клипа с запрошенной |
| `source_clip_timeout_seconds` | int | `1800` | Таймаут нарезки одного source-клипа (используется как нижняя граница вместе с `duration * 20`) |

## `output`

| Ключ | Тип | По умолчанию | Что делает |
|---|---|---|---|
| `root` | str | `/home/yuwye/tiktoks/output` | Куда складываются финальные `manifest.json` и `clip-XX/` |
| `keep_intermediates` | bool | `true` (`false` на хосте) | Если `false` — после завершения `process_video` удаляется весь `runtime.work_dir/<stream_id>` и временный каталог Remotion `public/jobs/<stream_id>` |
