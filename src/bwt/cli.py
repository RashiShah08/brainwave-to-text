"""Command-line interface.

    bwt info                     what data is present and what tasks exist
    bwt prepare                  epoch a task and cache it
    bwt train                    fit a model, evaluate it, write an artifact
    bwt evaluate                 score an existing pipeline, optionally permuted
    bwt benchmark                compare every pipeline on both protocols
    bwt predict FILE             decode a recording from the command line
    bwt models                   list trained artifacts
    bwt serve                    run the web service
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bwt import __version__
from bwt.logging_utils import get_logger, setup_logging
from bwt.paths import ensure_dirs, raw_data_dir, reports_dir

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _load_bundle(args):
    from bwt.data import load_bundle

    return load_bundle(
        args.task,
        dataset=getattr(args, "dataset", "eegmmidb"),
        tmin=args.tmin,
        tmax=args.tmax,
        n_jobs=args.n_jobs,
        use_cache=not args.no_cache,
        subjects=args.subjects,
    )


def _subject_list(value: str | None) -> list[int] | None:
    """Parse ``1,2,5-9`` into a subject list."""
    if not value:
        return None
    out: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def _evaluate(bundle, pipeline_name, protocols, splits, n_jobs, permutations):
    from bwt.evaluation import (
        cross_subject_cv, permutation_test, session_holdout, within_subject_cv,
    )
    from bwt.pipelines import pipeline_factory

    def factory():
        return pipeline_factory(
            pipeline_name, sfreq=bundle.sfreq, n_classes=len(bundle.classes)
        )()

    results = {}
    for protocol in protocols:
        log.info("running %s evaluation with %s", protocol, pipeline_name)
        if protocol == "within_subject":
            result = within_subject_cv(
                bundle, factory, pipeline_name=pipeline_name,
                n_splits=splits, n_jobs=n_jobs,
            )
        elif protocol == "cross_subject":
            result = cross_subject_cv(
                bundle, factory, pipeline_name=pipeline_name,
                n_splits=splits, n_jobs=1,
            )
        elif protocol == "session_holdout":
            result = session_holdout(
                bundle, factory, pipeline_name=pipeline_name,
            )
        else:
            raise ValueError(f"unknown protocol {protocol!r}")

        if permutations and protocol == "cross_subject":
            log.info("running %d-iteration permutation test", permutations)
            p_value, scores = permutation_test(
                bundle, factory, observed=result.mean_accuracy,
                n_permutations=permutations, n_splits=splits,
            )
            result.permutation_p = p_value
            result.permutation_scores = scores

        print("\n" + result.summary(), flush=True)
        results[protocol] = result
    return results


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_info(args) -> int:
    from bwt.data.physionet import (
        EXCLUDED_SUBJECTS, RUN_PROTOCOL, TASKS, available_subjects,
    )

    root = raw_data_dir()
    print(f"bwt {__version__}")
    print(f"raw data: {root}  ({'present' if root.is_dir() else 'MISSING'})")

    if root.is_dir():
        usable = available_subjects()
        every = available_subjects(include_excluded=True)
        print(f"subjects on disk: {len(every)}   usable: {len(usable)}")
        print(f"excluded: {sorted(EXCLUDED_SUBJECTS)} "
              "(128 Hz acquisition or corrupt annotations)")

    print("\nruns:")
    for run in sorted(RUN_PROTOCOL):
        spec = RUN_PROTOCOL[run]
        mapping = ", ".join(f"{k}={v.value}" for k, v in spec.annotation_map.items())
        print(f"  R{run:02d}  {spec.execution.value:9s}  {spec.description:38s}  {mapping}")

    print("\ntasks:")
    for name, task in TASKS.items():
        print(f"  {name:22s} runs={list(task.runs)}")
        print(f"  {'':22s} {task.description}")
        print(f"  {'':22s} classes={list(task.classes)}")
    return 0


def cmd_prepare(args) -> int:
    ensure_dirs()
    bundle = _load_bundle(args)
    print(bundle.summary())
    print(json.dumps(bundle.class_counts(), indent=2))
    return 0


def cmd_train(args) -> int:
    from bwt.artifacts import ModelCard, save_artifact
    from bwt.data.physionet import EXCLUDED_SUBJECTS, get_task
    from bwt.pipelines import TRANSDUCTIVE_PIPELINES, build_pipeline

    ensure_dirs()
    bundle = _load_bundle(args)
    task = get_task(args.task)
    print(bundle.summary())

    protocols = [] if args.no_eval else list(args.protocols)
    results = _evaluate(
        bundle, args.pipeline, protocols, args.cv_splits, args.n_jobs,
        args.permutations,
    )

    log.info("fitting final model on all %d trials", bundle.n_trials)
    model = build_pipeline(
        args.pipeline, sfreq=bundle.sfreq, n_classes=len(bundle.classes),
        random_state=args.random_state,
    )
    model.fit(bundle.X, bundle.y)

    counts = np.bincount(bundle.y, minlength=len(bundle.classes))
    card = ModelCard(
        name=f"{args.task}__{args.pipeline}",
        task=args.task,
        task_description=task.description,
        pipeline=args.pipeline,
        classes=list(bundle.classes),
        sfreq=bundle.sfreq,
        n_channels=bundle.n_channels,
        n_times=bundle.n_times,
        ch_names=list(bundle.ch_names),
        tmin=bundle.tmin,
        tmax=bundle.tmax,
        units=bundle.units,
        n_train_trials=bundle.n_trials,
        train_subjects=bundle.subjects,
        excluded_subjects=sorted(EXCLUDED_SUBJECTS),
        class_counts=bundle.class_counts(),
        evaluation={k: v.to_dict() for k, v in results.items()},
        chance_level=1.0 / len(bundle.classes),
        majority_level=float(counts.max() / counts.sum()),
        requires_batch_recentering=args.pipeline in TRANSDUCTIVE_PIPELINES,
        notes=args.notes or "",
    )

    path = save_artifact(model, card)
    print(f"\nartifact: {path}")
    print(f"performance: {card.headline()}")
    return 0


def cmd_evaluate(args) -> int:
    ensure_dirs()
    bundle = _load_bundle(args)
    print(bundle.summary())
    results = _evaluate(
        bundle, args.pipeline, list(args.protocols), args.cv_splits,
        args.n_jobs, args.permutations,
    )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {k: v.to_dict() for k, v in results.items()}, indent=2))
        print(f"\nwrote {path}")
    return 0


def cmd_benchmark(args) -> int:
    from bwt.pipelines import REGISTRY

    ensure_dirs()
    bundle = _load_bundle(args)
    print(bundle.summary())

    names = args.pipelines or sorted(REGISTRY)
    table: dict[str, dict] = {}
    for name in names:
        try:
            results = _evaluate(
                bundle, name, list(args.protocols), args.cv_splits,
                args.n_jobs, 0,
            )
            table[name] = {k: v.to_dict() for k, v in results.items()}
        except Exception as exc:  # noqa: BLE001
            log.error("%s failed: %s", name, exc)
            table[name] = {"error": str(exc)}

    out = Path(args.output or reports_dir() / f"benchmark_{args.task}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"task": args.task, "bundle": bundle.metadata(), "results": table}, indent=2))

    print(f"\n{'pipeline':22s} {'within-subject':>18s} {'cross-subject':>18s}")
    print("-" * 62)
    for name, entry in table.items():
        cells = []
        for protocol in ("within_subject", "cross_subject"):
            payload = entry.get(protocol, {})
            if "mean_accuracy" in payload:
                cells.append(
                    f"{payload['mean_accuracy']:.4f} +/-{payload['std_accuracy']:.3f}"
                )
            else:
                cells.append("-")
        print(f"{name:22s} {cells[0]:>18s} {cells[1]:>18s}")
    print(f"\nchance = {1.0 / len(bundle.classes):.4f};  wrote {out}")
    return 0


def cmd_predict(args) -> int:
    from bwt.serving.predictor import Predictor

    predictor = Predictor.load(args.model)
    batch = predictor.predict_edf(Path(args.file), spell=not args.no_spell)

    if args.json:
        print(json.dumps(
            {**batch.to_dict(), "performance": predictor.performance_note()},
            indent=2))
        return 0

    print(f"model     : {batch.model}")
    print(f"task      : {batch.task}   classes: {batch.classes}")
    print(f"epoching  : {batch.epoching}")
    print(f"epochs    : {batch.n}")
    for warning in batch.warnings:
        print(f"warning   : {warning}")
    print()
    for prediction in batch.predictions[: args.limit]:
        onset = "" if prediction.onset_seconds is None else f"t={prediction.onset_seconds:7.2f}s "
        print(f"  [{prediction.index:3d}] {onset}{prediction.label:12s} "
              f"p={prediction.confidence:.3f}")
    if batch.n > args.limit:
        print(f"  ... {batch.n - args.limit} more")
    print(f"\nmajority  : {batch.majority_label()} "
          f"(mean confidence {batch.mean_confidence():.3f})")
    if batch.text is not None:
        print(f"spelled   : {batch.text!r} "
              f"({batch.speller['trials_per_character']} trials/character)")
    return 0


def cmd_datasets(args) -> int:
    from bwt.data.datasets import get_dataset, list_datasets

    for name, description in list_datasets():
        print(f"{name}")
        print(f"  {description}")
        source = get_dataset(name) if name == "eegmmidb" else None
        if source is not None:
            try:
                print(f"  subjects present: {len(source.subjects())}")
            except Exception as exc:  # noqa: BLE001
                print(f"  subjects present: unknown ({exc})")
        else:
            print("  subjects: downloaded on first use via MOABB")
        tasks = get_dataset(name).tasks
        for task, classes in tasks.items():
            print(f"    {task:22s} {list(classes)}")
        print()
    return 0


def cmd_calibrate(args) -> int:
    from bwt.calibration import calibration_curve
    from bwt.pipelines import pipeline_factory

    ensure_dirs()
    bundle = _load_bundle(args)
    print(bundle.summary())

    def factory():
        return pipeline_factory(
            args.pipeline, sfreq=bundle.sfreq, n_classes=len(bundle.classes)
        )()

    curve = calibration_curve(
        bundle, factory, pipeline_name=args.pipeline,
        dataset=args.dataset, budgets=tuple(args.budgets),
        strategies=tuple(args.strategies), n_splits=args.repeats,
        finetune_epochs=args.finetune_epochs, max_subjects=args.max_subjects,
    )
    print("\n" + curve.summary())

    out = Path(args.output or reports_dir() /
               f"calibration_{args.dataset}_{args.task}_{args.pipeline}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(curve.to_dict(), indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


def cmd_stream(args) -> int:
    from bwt.serving.predictor import Predictor

    predictor = Predictor.load(args.model)
    stream = predictor.stream_from_edf(
        Path(args.file), speed=args.speed, step_seconds=args.step
    )
    decoder = predictor.streaming_decoder(
        threshold=args.threshold, max_windows=args.max_windows
    )

    print(f"model    : {predictor.name}")
    print(f"classes  : {predictor.card.classes}")
    print(f"windows  : {len(stream)} (step {args.step}s, speed {args.speed or 'max'})")
    print()

    decisions = timeouts = 0
    text = ""
    for event in decoder.run(stream):
        if event.decision is None:
            continue
        decision = event.decision
        if decision.timed_out:
            timeouts += 1
            print(f"  t={event.onset_seconds:7.2f}s  (timed out after "
                  f"{decision.n_windows} windows)")
        else:
            decisions += 1
            print(f"  t={event.onset_seconds:7.2f}s  {decision.label:12s} "
                  f"p={decision.confidence:.3f} after {decision.n_windows} windows")
        if event.speller:
            text = event.speller["text"]

    print(f"\ndecisions: {decisions} committed, {timeouts} timed out")
    print(f"spelled  : {text!r}")
    return 0


def cmd_explain(args) -> int:
    from bwt.artifacts import load_artifact, resolve_artifact
    from bwt.explain import ERD_WINDOW, csp_patterns, erd_curve, lateralisation_index

    ensure_dirs()
    out_dir = Path(args.output or reports_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    # ERD needs pre-cue samples, which the decoding window does not contain.
    args.tmin, args.tmax = ERD_WINDOW
    bundle = _load_bundle(args)
    print(bundle.summary())

    erd_path = erd_curve(bundle, output=out_dir / f"erd_{args.task}.png")
    print(f"ERD curve   -> {erd_path}")

    index = lateralisation_index(bundle)
    print("\nlateralisation index (C3 - C4, % change vs pre-cue baseline):")
    for name, value in index["index_per_class"].items():
        print(f"  {name:14s} {value:+.2f}")
    print(f"  {index['interpretation']}")

    try:
        path = resolve_artifact(args.model)
        model, card = load_artifact(path)
        csp_path = csp_patterns(model, card,
                                output=out_dir / f"csp_{card.task}.png")
        print(f"\nCSP patterns -> {csp_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nCSP patterns skipped: {exc}")

    summary = out_dir / f"explain_{args.task}.json"
    summary.write_text(json.dumps(index, indent=2, default=float))
    print(f"wrote {summary}")
    return 0


def cmd_models(args) -> int:
    from bwt.artifacts import list_artifacts

    found = list_artifacts()
    if not found:
        print("no trained models. Run `bwt train` first.")
        return 1
    for path, card in found:
        print(f"{path.name}")
        print(f"  task     {card.task}  classes={card.classes}")
        print(f"  created  {card.created_utc}")
        print(f"  input    {card.n_channels}ch x {card.n_times} @ {card.sfreq:g} Hz "
              f"({card.tmin}-{card.tmax}s, {card.units})")
        print(f"  result   {card.headline()}")
        print()
    return 0


def cmd_serve(args) -> int:  # pragma: no cover - long running
    from waitress import serve

    from bwt.config import Config
    from bwt.serving.app import create_app

    config = Config.load()
    if args.model:
        config.serve.model = args.model
    if args.host:
        config.serve.host = args.host
    if args.port:
        config.serve.port = args.port

    app = create_app(config)
    if args.dev:
        log.warning(
            "development server requested; debug remains OFF because the "
            "Werkzeug debugger permits remote code execution"
        )
        app.run(host=config.serve.host, port=config.serve.port, debug=False)
    else:
        log.info("serving on http://%s:%d", config.serve.host, config.serve.port)
        serve(app, host=config.serve.host, port=config.serve.port, threads=4)
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    from bwt.data.datasets import DEFAULT_DATASET, list_datasets
    from bwt.data.epochs import DEFAULT_TMAX, DEFAULT_TMIN
    from bwt.data.physionet import DEFAULT_TASK, TASKS
    from bwt.pipelines import DEFAULT_PIPELINE, REGISTRY

    parser = argparse.ArgumentParser(
        prog="bwt",
        description="Brainwave-to-Text: EEG motor-imagery decoding and spelling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"bwt {__version__}")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p):
        p.add_argument("--dataset", default=DEFAULT_DATASET,
                       choices=[n for n, _ in list_datasets()])
        p.add_argument("--task", default=DEFAULT_TASK,
                       help=f"one of {sorted(TASKS)} for eegmmidb; "
                            "bnci2a supports mi_left_right and mi_four_class")
        p.add_argument("--tmin", type=float, default=DEFAULT_TMIN)
        p.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
        p.add_argument("--subjects", type=_subject_list, default=None,
                       help="subset such as '1,2,5-9'; default is all usable")
        p.add_argument("--n-jobs", type=int, default=4)
        p.add_argument("--no-cache", action="store_true")

    def add_cv_args(p):
        p.add_argument("--protocols", nargs="+",
                       default=["within_subject", "cross_subject"],
                       choices=["within_subject", "cross_subject",
                                "session_holdout"],
                       help="session_holdout needs a dataset with sessions "
                            "(bnci2a): train on day one, test on day two")
        p.add_argument("--cv-splits", type=int, default=5)
        p.add_argument("--permutations", type=int, default=0,
                       help="permutation-test iterations (0 = skip)")

    p = sub.add_parser("info", help="describe the dataset and available tasks")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("prepare", help="epoch a task and populate the cache")
    add_data_args(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="fit, evaluate, and save a model")
    add_data_args(p)
    add_cv_args(p)
    p.add_argument("--pipeline", default=DEFAULT_PIPELINE, choices=sorted(REGISTRY))
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--no-eval", action="store_true",
                   help="skip cross-validation (the model card will say so)")
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="cross-validate without saving a model")
    add_data_args(p)
    add_cv_args(p)
    p.add_argument("--pipeline", default=DEFAULT_PIPELINE, choices=sorted(REGISTRY))
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("benchmark", help="compare all pipelines")
    add_data_args(p)
    add_cv_args(p)
    p.add_argument("--pipelines", nargs="+", default=None, choices=sorted(REGISTRY))
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("predict", help="decode one EDF recording")
    p.add_argument("file")
    p.add_argument("--model", default=None, help="artifact name or path")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-spell", action="store_true")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("datasets", help="list datasets and their tasks")
    p.set_defaults(func=cmd_datasets)

    p = sub.add_parser("calibrate",
                       help="measure accuracy vs number of calibration trials")
    add_data_args(p)
    p.add_argument("--pipeline", default="eegnet", choices=sorted(REGISTRY))
    p.add_argument("--budgets", type=int, nargs="+", default=[0, 5, 10, 20, 40],
                   help="calibration trial counts to sweep")
    p.add_argument("--strategies", nargs="+", default=["none", "finetune"],
                   choices=["none", "finetune", "refit"])
    p.add_argument("--repeats", type=int, default=3,
                   help="random calibration subsets per budget")
    p.add_argument("--finetune-epochs", type=int, default=40)
    p.add_argument("--max-subjects", type=int, default=15,
                   help="subjects to evaluate (each needs a population model)")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("stream",
                       help="decode a recording continuously with evidence accumulation")
    p.add_argument("file")
    p.add_argument("--model", default=None)
    p.add_argument("--speed", type=float, default=0.0,
                   help="replay speed; 1.0 is real time, 0 is as fast as possible")
    p.add_argument("--step", type=float, default=0.5, help="window step in seconds")
    p.add_argument("--threshold", type=float, default=0.9,
                   help="posterior required to commit a decision")
    p.add_argument("--max-windows", type=int, default=40)
    p.set_defaults(func=cmd_stream)

    p = sub.add_parser("explain",
                       help="ERD curves, CSP topographies, lateralisation index")
    add_data_args(p)
    p.add_argument("--model", default=None)
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_explain)

    p = sub.add_parser("models", help="list trained artifacts")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("serve", help="run the web service")
    p.add_argument("--model", default=None)
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--dev", action="store_true",
                   help="use the Flask dev server instead of waitress")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        log.error("%s: %s", type(exc).__name__, exc)
        if args.log_level == "DEBUG":
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
