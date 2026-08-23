"""Command-line entry point for reproducible medical MT evaluation runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from medmt_eval.data.io import load_segments, read_rows, write_jsonl, write_parquet
from medmt_eval.inference.runner import evaluate_hypotheses, translate_segments
from medmt_eval.metrics.neural import CometScorer
from medmt_eval.metrics.surface import corpus_metric
from medmt_eval.models.factory import create_translator
from medmt_eval.report.summary import aggregate_evaluations, plot_divergence, write_master_table
from medmt_eval.schema import Segment
from medmt_eval.glossary import Glossary
from medmt_eval.stats.bootstrap import mcnemar_exact, paired_bootstrap_metric
from medmt_eval.taxonomy.clinical import ClinicalSafetyEvaluator, TerminologyBank


def _save_json(payload: dict[str, Any], path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return file_path


def _segments_and_hypotheses(args: argparse.Namespace) -> tuple[list[Segment], list[str], list[dict[str, Any]]]:
    raw_rows = read_rows(args.input)
    segments = load_segments(
        args.input,
        default_src_lang=args.src_lang,
        default_tgt_lang=args.tgt_lang,
        reverse=args.reverse,
    )
    aliases = ("hyp_text", "hypothesis", "translation", "mt", "prediction")
    hypotheses: list[str] = []
    for index, row in enumerate(raw_rows, start=1):
        value = next((row.get(key) for key in aliases if row.get(key) is not None), None)
        if value is None:
            raise ValueError(f"Input row {index} has no hypothesis; expected one of {', '.join(aliases)}.")
        hypotheses.append(str(value))
    if args.reverse:
        # A reversed evaluation needs predictions in the reversed direction too.
        raise ValueError("--reverse is supported for translation inputs, not existing hypotheses.")
    return segments, hypotheses, raw_rows


def _safety_evaluator(term_bank_path: str | None) -> ClinicalSafetyEvaluator:
    return ClinicalSafetyEvaluator(TerminologyBank.from_csv(term_bank_path) if term_bank_path else None)


def _comet_from_args(args: argparse.Namespace) -> CometScorer | None:
    checkpoint = getattr(args, "comet_checkpoint", None)
    return None if not checkpoint else CometScorer(checkpoint, reference_free=args.comet_reference_free)


def _write_evaluations(evaluations, output: str, parquet: str | None) -> None:
    rows = [evaluation.to_dict() for evaluation in evaluations]
    write_jsonl(rows, output)
    if parquet:
        write_parquet(rows, parquet)


def command_translate(args: argparse.Namespace) -> int:
    segments = load_segments(
        args.input,
        default_src_lang=args.src_lang,
        default_tgt_lang=args.tgt_lang,
        reverse=args.reverse,
    )
    translator = create_translator(
        args.model,
        model_id=args.model_id,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        prompt_template=getattr(args, "prompt_template", None),
        api_key=getattr(args, "api_key", None),
        free_tier=not getattr(args, "paid_tier", False),
    )
    output = translate_segments(
        translator, segments,
        chunk_max_tokens=getattr(args, "chunk_max_tokens", 0) or None,
    )
    write_jsonl(output, args.output)
    if args.parquet:
        write_parquet(output, args.parquet)
    print(json.dumps({"output": args.output, "model": translator.name, "n_segments": len(output)}))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    segments, hypotheses, raw_rows = _segments_and_hypotheses(args)
    model = args.model or str(raw_rows[0].get("model", "unknown"))
    evaluations, summary = evaluate_hypotheses(
        segments,
        hypotheses,
        model=model,
        safety_evaluator=_safety_evaluator(args.term_bank),
        comet=_comet_from_args(args),
        comet_batch_size=args.comet_batch_size,
        comet_gpus=args.comet_gpus,
    )
    _write_evaluations(evaluations, args.output, args.parquet)
    _save_json(summary, args.summary or f"{args.output}.summary.json")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def command_run(args: argparse.Namespace) -> int:
    segments = load_segments(
        args.input,
        default_src_lang=args.src_lang,
        default_tgt_lang=args.tgt_lang,
        reverse=args.reverse,
    )
    translator = create_translator(
        args.model,
        model_id=args.model_id,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        prompt_template=getattr(args, "prompt_template", None),
        api_key=getattr(args, "api_key", None),
        free_tier=not getattr(args, "paid_tier", False),
    )
    predictions = translate_segments(
        translator, segments,
        chunk_max_tokens=getattr(args, "chunk_max_tokens", 0) or None,
    )
    if args.predictions_output:
        write_jsonl(predictions, args.predictions_output)
    evaluations, summary = evaluate_hypotheses(
        segments,
        [str(row["hyp_text"]) for row in predictions],
        model=translator.name,
        # Take the generation record from the predictions, not the translator:
        # translate_segments adds the `chunked`/`chunk_max_tokens` markers there,
        # and reading it off the translator would silently drop them.
        generation=predictions[0].get("generation") if predictions else translator.generation_config,
        safety_evaluator=_safety_evaluator(args.term_bank),
        comet=_comet_from_args(args),
        comet_batch_size=args.comet_batch_size,
        comet_gpus=args.comet_gpus,
    )
    _write_evaluations(evaluations, args.output, args.parquet)
    _save_json(summary, args.summary or f"{args.output}.summary.json")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def command_report(args: argparse.Namespace) -> int:
    rows = read_rows(args.input)
    table = aggregate_evaluations(rows)
    output_dir = Path(args.output_dir)
    table_path = write_master_table(table, output_dir / "master_metrics.csv")
    result: dict[str, str] = {"master_table": str(table_path)}
    if not args.no_plot:
        result["divergence_plot"] = str(plot_divergence(table, output_dir / "bleu_vs_critical_error.png"))
    _save_json({"table": table}, output_dir / "master_metrics.json")
    print(json.dumps(result))
    return 0


def _comparison_data(
    path: str, src_lang: str | None, tgt_lang: str | None
) -> dict[str, tuple[str, str, bool | None]]:
    raw_rows = read_rows(path)
    segments = load_segments(path, default_src_lang=src_lang, default_tgt_lang=tgt_lang)
    aliases = ("hyp_text", "hypothesis", "translation", "mt", "prediction")
    output: dict[str, tuple[str, str, bool | None]] = {}
    for segment, row in zip(segments, raw_rows):
        hypothesis = next((row.get(key) for key in aliases if row.get(key) is not None), None)
        if hypothesis is None or segment.ref_text is None:
            raise ValueError(f"Comparison input {path} needs hyp_text and ref_text for every row.")
        if segment.id in output:
            raise ValueError(f"Duplicate segment ID {segment.id!r} in {path}.")
        critical = bool(row["has_critical_error"]) if "has_critical_error" in row else None
        output[segment.id] = (str(hypothesis), str(segment.ref_text), critical)
    return output


def command_compare(args: argparse.Namespace) -> int:
    baseline = _comparison_data(args.baseline, args.src_lang, args.tgt_lang)
    candidate = _comparison_data(args.candidate, args.src_lang, args.tgt_lang)
    if baseline.keys() != candidate.keys():
        raise ValueError("Baseline and candidate must contain exactly the same segment IDs.")
    ids = sorted(baseline)
    references = [baseline[segment_id][1] for segment_id in ids]
    if references != [candidate[segment_id][1] for segment_id in ids]:
        raise ValueError("Baseline and candidate references differ; compare against a single gold set.")
    metric_name = args.metric.lower()
    higher_is_better = metric_name != "ter"

    def quality_metric(hypotheses: Sequence[str], refs: Sequence[str]) -> float:
        raw = corpus_metric(metric_name, hypotheses, refs)
        return raw if higher_is_better else -raw

    result = paired_bootstrap_metric(
        [baseline[segment_id][0] for segment_id in ids],
        [candidate[segment_id][0] for segment_id in ids],
        references,
        quality_metric,
        resamples=args.resamples,
        seed=args.seed,
    ).to_dict()
    result.update({"metric": metric_name, "higher_is_better": higher_is_better, "n_segments": len(ids)})
    if args.error_incidence:
        if any(baseline[segment_id][2] is None or candidate[segment_id][2] is None for segment_id in ids):
            raise ValueError(
                "--error-incidence requires evaluated inputs containing has_critical_error for every segment."
            )
        result["mcnemar_critical_error"] = mcnemar_exact(
            [bool(baseline[segment_id][2]) for segment_id in ids],
            [bool(candidate[segment_id][2]) for segment_id in ids],
        )
    if args.output:
        _save_json(result, args.output)
    print(json.dumps(result))
    return 0


def command_leaderboard(args: argparse.Namespace) -> int:
    """Aggregate evaluations from multiple systems into a unified leaderboard."""
    all_rows: list[dict[str, Any]] = []
    for input_path in args.inputs:
        all_rows.extend(read_rows(input_path))
    if not all_rows:
        raise ValueError("No evaluation rows found in the provided inputs.")
    table = aggregate_evaluations(all_rows)
    output_dir = Path(args.output_dir)
    table_path = write_master_table(table, output_dir / "leaderboard.csv")
    result: dict[str, str] = {"leaderboard_table": str(table_path)}
    if not args.no_plot:
        result["divergence_plot"] = str(plot_divergence(table, output_dir / "bleu_vs_critical_error.png"))
    _save_json({"table": table}, output_dir / "leaderboard.json")
    print(json.dumps(result))
    return 0


def command_convert_himl(args: argparse.Namespace) -> int:
    """Convert HimL SGML test sets to normalized JSONL."""
    from medmt_eval.data.himl_sgm import load_himl_from_tar

    segments = load_himl_from_tar(
        args.input,
        year=args.year,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
    )
    rows = [segment.to_dict() for segment in segments]
    write_jsonl(rows, args.output)
    print(json.dumps({"output": args.output, "n_segments": len(rows), "year": args.year}))
    return 0


def command_convert_emea(args: argparse.Namespace) -> int:
    """Convert EMEA TMX to normalized JSONL with alignment filtering."""
    from medmt_eval.data.tmx import load_emea_from_tmx

    sample_size = args.sample_size if args.sample_size > 0 else None
    segments = load_emea_from_tmx(
        args.input,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        min_length=args.min_length,
        max_length_ratio=args.max_length_ratio,
        sample_size=sample_size,
        seed=args.seed,
    )
    rows = [segment.to_dict() for segment in segments]
    write_jsonl(rows, args.output)
    print(json.dumps({"output": args.output, "n_segments": len(rows)}))
    return 0


def command_convert_parrot(args: argparse.Namespace) -> int:
    """Convert the German subset of PARROT radiology reports to normalized JSONL."""
    from medmt_eval.data.parrot import parrot_rows

    rows = parrot_rows(
        args.input,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        sections=tuple(args.sections) if getattr(args, "sections", None) else None,
        sections_mode=getattr(args, "sections_mode", "lenient"),
    )
    write_jsonl(rows, args.output)
    print(json.dumps({"output": args.output, "n_segments": len(rows)}))
    return 0


def _subsample(segments, size: int, seed: int):
    """Deterministic, length-stratified subsample shared by the experiments."""
    if not size or size >= len(segments):
        return segments
    import random

    ordered = sorted(segments, key=lambda s: len(s.src_text))
    stride = len(ordered) / size
    picked = [ordered[int(i * stride)] for i in range(size)]
    random.Random(seed).shuffle(picked)
    return picked


def _load_experiment_glossary(args: argparse.Namespace):
    """Load, domain-filter and split the glossary for an experiment run."""
    from medmt_eval.glossary import Glossary

    glossary = Glossary.from_csv(args.glossary)
    if not getattr(args, "allow_mt_derived", False):
        glossary = glossary.exclude_mt()
    if getattr(args, "branches", None):
        glossary = glossary.in_branches(args.branches)
    held_out = None
    if getattr(args, "holdout", 0.0):
        # Injecting the terms that are later scored would improve the
        # terminology detector by construction. Split so the two are disjoint.
        glossary, held_out = glossary.split(holdout=args.holdout, seed=args.seed)
    return glossary, held_out


def command_glossary_run(args: argparse.Namespace) -> int:
    """Experiment 1 — translate with and without an injected glossary."""
    from medmt_eval.inference.glossary_mt import run_glossary_experiment, summarise_by_arm
    from medmt_eval.models.chat import create_chat_backend

    segments = load_segments(
        args.input, default_src_lang=args.src_lang,
        default_tgt_lang=args.tgt_lang, reverse=args.reverse,
    )
    segments = _subsample(segments, args.sample_size, args.seed)
    glossary, held_out = _load_experiment_glossary(args)

    backend = create_chat_backend(
        args.backend, api_key=getattr(args, "api_key", None),
        max_tokens=args.max_new_tokens, device=args.device,
    )
    rows = run_glossary_experiment(
        backend, segments, glossary,
        arms=args.arms, limit=args.max_terms, seed=args.seed,
        safety_evaluator=_safety_evaluator(args.term_bank),
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    write_jsonl(rows, args.output)
    summary = summarise_by_arm(rows)
    _save_json(
        {"arms": summary,
         "glossary_injected": len(glossary),
         "glossary_held_out": len(held_out) if held_out is not None else 0},
        args.summary or f"{args.output}.summary.json",
    )
    print(json.dumps({"output": args.output, "n_rows": len(rows),
                      "n_segments": len(segments), "arms": list(args.arms)}))
    return 0


def command_debate(args: argparse.Namespace) -> int:
    """Experiment 2 — three personas argue about the translation and converge."""
    from medmt_eval.inference.debate import PERSONAS, Agent, run_debate, summarise_debate
    from medmt_eval.models.chat import create_chat_backend

    segments = load_segments(
        args.input, default_src_lang=args.src_lang,
        default_tgt_lang=args.tgt_lang, reverse=args.reverse,
    )
    segments = _subsample(segments, args.sample_size, args.seed)
    glossary, _ = _load_experiment_glossary(args)

    names = list(PERSONAS)
    if len(args.backends) != len(names):
        raise SystemExit(
            f"--backends needs exactly {len(names)} entries, one per persona "
            f"({', '.join(names)}); got {len(args.backends)}."
        )
    # Each persona gets a different slice of the glossary, so the agents disagree
    # for substantive reasons rather than by sampling noise.
    slices = _split_evenly(glossary, len(names), args.seed)
    agents = [
        Agent(
            name=name,
            backend=create_chat_backend(
                spec, api_key=getattr(args, "api_key", None),
                max_tokens=args.max_new_tokens, device=args.device,
            ),
            persona=PERSONAS[name],
            glossary=piece,
        )
        for name, spec, piece in zip(names, args.backends, slices)
    ]
    synthesiser = (
        create_chat_backend(args.synthesiser, api_key=getattr(args, "api_key", None),
                            max_tokens=args.max_new_tokens, device=args.device)
        if args.synthesiser else None
    )

    rows = run_debate(
        agents, segments, rounds=args.rounds, limit=args.max_terms,
        synthesiser=synthesiser,
        safety_evaluator=_safety_evaluator(args.term_bank),
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    write_jsonl(rows, args.output)
    _save_json({"variants": summarise_debate(rows)},
               args.summary or f"{args.output}.summary.json")
    print(json.dumps({"output": args.output, "n_segments": len(rows),
                      "rounds": args.rounds,
                      "agents": {n: b for n, b in zip(names, args.backends)}}))
    return 0


def _split_evenly(glossary, parts: int, seed: int):
    """Deal the glossary into `parts` disjoint slices, one per agent."""
    from medmt_eval.glossary import Glossary

    entries = sorted(glossary, key=lambda e: e.concept_id)
    import random

    random.Random(seed).shuffle(entries)
    return [Glossary(entries[i::parts]) for i in range(parts)]


def command_cascade(args: argparse.Namespace) -> int:
    """Sequential review pipeline: translate -> terminologise -> arbitrate."""
    from medmt_eval.inference.cascade import (
        PERSONAS, ROLES, Stage, run_cascade, summarise_cascade,
    )
    from medmt_eval.models.chat import create_chat_backend

    segments = load_segments(
        args.input, default_src_lang=args.src_lang,
        default_tgt_lang=args.tgt_lang, reverse=args.reverse,
    )
    segments = _subsample(segments, args.sample_size, args.seed)
    glossary, _ = _load_experiment_glossary(args)

    if len(args.backends) != len(args.roles):
        raise SystemExit(
            f"--backends and --roles must have the same length "
            f"({len(args.backends)} vs {len(args.roles)})."
        )
    # Only the stage named by --glossary-stage is given the dictionary. The whole
    # design rests on the terminologist having it and the arbiter not.
    stages = [
        Stage(
            name=f"{index + 1}-{role}",
            role=role,
            backend=create_chat_backend(
                spec, api_key=getattr(args, "api_key", None),
                max_tokens=args.max_new_tokens, device=args.device,
            ),
            glossary=glossary if role == args.glossary_stage else Glossary(),
            max_terms=args.max_terms,
        )
        for index, (spec, role) in enumerate(zip(args.backends, args.roles))
    ]

    rows = run_cascade(
        stages, segments,
        safety_evaluator=_safety_evaluator(args.term_bank),
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    write_jsonl(rows, args.output)
    _save_json({"stages": summarise_cascade(rows)},
               args.summary or f"{args.output}.summary.json")
    print(json.dumps({"output": args.output, "n_segments": len(rows),
                      "stages": [s.name for s in stages]}))
    return 0


def command_judge(args: argparse.Namespace) -> int:
    """Audit existing translations with an LLM clinical rubric."""
    from medmt_eval.metrics.llm_judge import (
        agreement_with_detectors, run_llm_judge, summarise_judge,
    )
    from medmt_eval.models.chat import create_chat_backend

    rows = [json.loads(line) for line in open(args.input, encoding="utf-8") if line.strip()]
    if args.sample_size and args.sample_size < len(rows):
        rows = rows[: args.sample_size]
    backend = create_chat_backend(
        args.judge, api_key=getattr(args, "api_key", None),
        max_tokens=args.max_new_tokens, device=args.device,
    )
    judged = run_llm_judge(
        backend, rows,
        src_key=args.src_key, hyp_key=args.hyp_key,
        src_lang=args.src_lang, tgt_lang=args.tgt_lang,
        system_under_test=args.system_under_test,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    write_jsonl(judged, args.output)
    _save_json(
        {"judge": summarise_judge(judged),
         "vs_rule_detectors": agreement_with_detectors(judged)},
        args.summary or f"{args.output}.summary.json",
    )
    print(json.dumps({"output": args.output, "n": len(judged)}))
    return 0


def command_roundtrip(args: argparse.Namespace) -> int:
    """Iteratively translate back and forth, scoring every step against fixed anchors."""
    from medmt_eval.inference.roundtrip import run_roundtrip, summarise_by_step

    segments = load_segments(
        args.input,
        default_src_lang=args.src_lang,
        default_tgt_lang=args.tgt_lang,
        reverse=args.reverse,
    )
    if args.sample_size and args.sample_size < len(segments):
        # Deterministic, length-stratified subsample: 20 translation passes over
        # the full corpus is far beyond the wall-clock limit for larger models.
        import random

        ordered = sorted(segments, key=lambda s: len(s.src_text))
        stride = len(ordered) / args.sample_size
        segments = [ordered[int(i * stride)] for i in range(args.sample_size)]
        random.Random(args.seed).shuffle(segments)

    def factory(src: str, tgt: str):
        # A fresh instance per direction: Opus ships one checkpoint per pair and
        # refuses to switch direction on a live instance.
        return create_translator(
            args.model,
            model_id=args.model_id,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            prompt_template=getattr(args, "prompt_template", None),
            api_key=getattr(args, "api_key", None),
            free_tier=not getattr(args, "paid_tier", False),
        )

    rows = run_roundtrip(
        factory,
        segments,
        cycles=args.cycles,
        safety_evaluator=_safety_evaluator(args.term_bank),
        chunk_max_tokens=getattr(args, "chunk_max_tokens", 0) or None,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )
    write_jsonl(rows, args.output)
    summary = summarise_by_step(rows)
    _save_json({"steps": summary}, args.summary or f"{args.output}.summary.json")
    print(json.dumps({"output": args.output, "n_rows": len(rows),
                      "n_segments": len(segments), "cycles": args.cycles}))
    return 0


_ARMS = ("none", "glossary", "distractor")
_CASCADE_ROLES = ("translate", "terminologise", "arbitrate")


def _add_data_args(parser: argparse.ArgumentParser, *, allow_reverse: bool = True) -> None:
    parser.add_argument("--input", required=True, help="JSONL, CSV, TSV, or Parquet input")
    parser.add_argument("--src-lang", help="Default source language when input omits it (en/de)")
    parser.add_argument("--tgt-lang", help="Default target language when input omits it (en/de)")
    if allow_reverse:
        parser.add_argument("--reverse", action="store_true", help="Reverse source/reference for DE↔EN evaluation")


_MODEL_CHOICES = ["identity", "opus", "nllb", "madlad", "tower", "deepl", "prompted-llm", "hymt2", "translategemma", "openai-compat"]


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=_MODEL_CHOICES)
    parser.add_argument("--model-id", help="Override the adapter's default Hugging Face model ID")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", help="Torch device, for example cuda or cpu")
    parser.add_argument(
        "--chunk-max-tokens", type=int, default=0,
        help="Split each segment on sentence boundaries into chunks of at most N "
             "tokens, translate separately, then reassemble before scoring. Needed "
             "for models with a hard encoder limit (e.g. NLLB at 512). 0 disables.",
    )
    parser.add_argument("--prompt-template", help="Custom prompt template for prompted-llm adapter")
    parser.add_argument("--api-key", help="API key for deepl or hosted-llm adapters")
    parser.add_argument(
        "--paid-tier", action="store_true",
        help="Use the paid DeepL API endpoint instead of the free tier",
    )


def _add_scoring_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--term-bank", help="CSV containing concept_id,en,de exact term pairs")
    parser.add_argument("--comet-checkpoint", help="Optional COMET/XCOMET checkpoint name")
    parser.add_argument("--comet-reference-free", action="store_true", help="Use a QE checkpoint without refs")
    parser.add_argument("--comet-batch-size", type=int, default=8)
    parser.add_argument("--comet-gpus", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medmt-eval", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    translate = commands.add_parser("translate", help="Translate a normalized corpus")
    _add_data_args(translate)
    _add_model_args(translate)
    translate.add_argument("--output", required=True, help="Prediction JSONL output")
    translate.add_argument("--parquet", help="Optional prediction Parquet output")
    translate.set_defaults(handler=command_translate)

    evaluate = commands.add_parser("evaluate", help="Score existing hypotheses")
    _add_data_args(evaluate, allow_reverse=False)
    evaluate.add_argument("--model", help="Model label; defaults to input's model field")
    _add_scoring_args(evaluate)
    evaluate.add_argument("--output", required=True, help="Per-segment evaluation JSONL")
    evaluate.add_argument("--parquet", help="Optional per-segment evaluation Parquet")
    evaluate.add_argument("--summary", help="Summary JSON (default: OUTPUT.summary.json)")
    evaluate.set_defaults(handler=command_evaluate, reverse=False)

    run = commands.add_parser("run", help="Translate then evaluate in one reproducible command")
    _add_data_args(run)
    _add_model_args(run)
    _add_scoring_args(run)
    run.add_argument("--output", required=True, help="Per-segment evaluation JSONL")
    run.add_argument("--predictions-output", help="Optional unscored translation JSONL")
    run.add_argument("--parquet", help="Optional per-segment evaluation Parquet")
    run.add_argument("--summary", help="Summary JSON (default: OUTPUT.summary.json)")
    run.set_defaults(handler=command_run)

    report = commands.add_parser("report", help="Build the master table and divergence plot")
    report.add_argument("--input", required=True, help="Per-segment evaluation JSONL/CSV/Parquet")
    report.add_argument("--output-dir", required=True)
    report.add_argument("--no-plot", action="store_true", help="Write tables only")
    report.set_defaults(handler=command_report)

    compare = commands.add_parser("compare", help="Paired bootstrap comparison of two systems")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--src-lang", help="Defaults if comparison input omits languages")
    compare.add_argument("--tgt-lang", help="Defaults if comparison input omits languages")
    compare.add_argument("--metric", choices=["bleu", "chrf", "ter"], default="chrf")
    compare.add_argument("--resamples", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=13)
    compare.add_argument("--error-incidence", action="store_true", help="Also calculate exact McNemar test")
    compare.add_argument("--output", help="Optional comparison JSON")
    compare.set_defaults(handler=command_compare)

    # --- Leaderboard: multi-system comparison ---
    leaderboard = commands.add_parser(
        "leaderboard",
        help="Aggregate evaluations from multiple systems into a single leaderboard table",
    )
    leaderboard.add_argument(
        "--inputs", required=True, nargs="+",
        help="Per-segment evaluation JSONL files (one per system, each must contain a 'model' column)",
    )
    leaderboard.add_argument("--output-dir", required=True, help="Directory for leaderboard outputs")
    leaderboard.add_argument("--no-plot", action="store_true", help="Skip the divergence plot")
    leaderboard.set_defaults(handler=command_leaderboard)

    # --- Data converters ---
    convert = commands.add_parser(
        "convert",
        help="Convert benchmark data (HimL SGML, EMEA TMX) into the normalized JSONL schema",
    )
    convert_sub = convert.add_subparsers(dest="convert_command", required=True)

    himl = convert_sub.add_parser("himl", help="Convert HimL SGML test sets to JSONL")
    himl.add_argument("--input", required=True, help="Path to himl-test-2015.tgz or himl-test-2017.tgz")
    himl.add_argument("--year", type=int, choices=[2015, 2017], required=True)
    himl.add_argument("--output", required=True, help="Output JSONL path")
    himl.add_argument("--src-lang", default="en")
    himl.add_argument("--tgt-lang", default="de")
    himl.set_defaults(handler=command_convert_himl)

    emea = convert_sub.add_parser("emea", help="Convert EMEA TMX to JSONL with filtering")
    emea.add_argument("--input", required=True, help="Path to emea-de-en.tmx.gz")
    emea.add_argument("--output", required=True, help="Output JSONL path")
    emea.add_argument("--src-lang", default="de")
    emea.add_argument("--tgt-lang", default="en")
    emea.add_argument("--min-length", type=int, default=3, help="Min character length per side")
    emea.add_argument("--max-length-ratio", type=float, default=3.0, help="Max len(longer)/len(shorter)")
    emea.add_argument(
        "--sample-size", type=int, default=400,
        help="Target number of segments after filtering (0 = keep all)",
    )
    emea.add_argument("--seed", type=int, default=13, help="RNG seed for deterministic sampling")
    emea.set_defaults(handler=command_convert_emea)

    parrot = convert_sub.add_parser(
        "parrot",
        help="Convert the German subset of PARROT radiology reports to JSONL",
    )
    parrot.add_argument("--input", required=True, help="Path to PARROT_v1_0.jsonl")
    parrot.add_argument("--output", required=True, help="Output JSONL path")
    parrot.add_argument(
        "--src-lang", default="de",
        help="Report language as source: de (default) or tr. Use en to invert the "
             "pair (English translation as source, original report as reference).",
    )
    parrot.add_argument("--tgt-lang", default="en")
    parrot.add_argument(
        "--sections", nargs="+", default=None,
        choices=["indication", "technique", "findings", "impression", "other"],
        help="Reduce each report to these sections. 'findings impression' is the "
             "medical text; the acquisition preamble contributes ~30%% of critical "
             "findings as protocol parameters rather than patient facts.",
    )
    parrot.add_argument(
        "--sections-mode", default="lenient", choices=["lenient", "strict"],
        help="lenient: keep the whole report when it has no such section "
             "(default). strict: drop it. Only 109 of 296 German reports carry "
             "the headers on both sides, so the two modes give very different "
             "corpora and are not comparable.",
    )
    parrot.set_defaults(handler=command_convert_parrot)

    glossary_run = commands.add_parser(
        "glossary-run",
        help="Experiment 1: translate with no glossary, the right glossary, and a distractor glossary",
    )
    _add_data_args(glossary_run)
    _add_scoring_args(glossary_run)
    glossary_run.add_argument("--backend", required=True,
                              help="api:<model> or local:<hf-id>, e.g. api:glm-5.2")
    glossary_run.add_argument("--glossary", required=True, help="Glossary CSV")
    glossary_run.add_argument("--arms", nargs="+", default=list(_ARMS),
                              choices=list(_ARMS))
    glossary_run.add_argument("--branches", default="ACE",
                              help="MeSH tree branches to keep (default ACE: anatomy, disease, diagnostics)")
    glossary_run.add_argument("--holdout", type=float, default=0.0,
                              help="Fraction of concepts withheld from injection so the "
                                   "terminology metric is not scored on injected terms")
    glossary_run.add_argument("--allow-mt-derived", action="store_true",
                              help="Keep machine-translated terminologies (German MeSH)")
    glossary_run.add_argument("--max-terms", type=int, default=40)
    glossary_run.add_argument("--sample-size", type=int, default=0)
    glossary_run.add_argument("--seed", type=int, default=13)
    glossary_run.add_argument("--api-key")
    glossary_run.add_argument("--device")
    glossary_run.add_argument("--max-new-tokens", type=int, default=2048)
    glossary_run.add_argument("--output", required=True)
    glossary_run.add_argument("--summary")
    glossary_run.set_defaults(handler=command_glossary_run)

    debate = commands.add_parser(
        "debate",
        help="Experiment 2: three personas with different glossaries argue and converge",
    )
    _add_data_args(debate)
    _add_scoring_args(debate)
    debate.add_argument("--backends", nargs="+", required=True,
                        help="One backend per persona (anatomist safety linguist), "
                             "e.g. api:glm-5.2 api:DeepSeek-V4-Flash api:MiniMax-M3")
    debate.add_argument("--synthesiser", help="Backend for the final sign-off "
                                              "(default: the first agent's)")
    debate.add_argument("--glossary", required=True)
    debate.add_argument("--branches", default="ACE")
    debate.add_argument("--holdout", type=float, default=0.0)
    debate.add_argument("--allow-mt-derived", action="store_true")
    debate.add_argument("--rounds", type=int, default=2)
    debate.add_argument("--max-terms", type=int, default=40)
    debate.add_argument("--sample-size", type=int, default=0)
    debate.add_argument("--seed", type=int, default=13)
    debate.add_argument("--api-key")
    debate.add_argument("--device")
    debate.add_argument("--max-new-tokens", type=int, default=2048)
    debate.add_argument("--output", required=True)
    debate.add_argument("--summary")
    debate.set_defaults(handler=command_debate)

    cascade = commands.add_parser(
        "cascade",
        help="Sequential pipeline: an MT model drafts, a terminologist corrects, an expert arbitrates",
    )
    _add_data_args(cascade)
    _add_scoring_args(cascade)
    cascade.add_argument("--backends", nargs="+", required=True,
                         help="One backend per stage, in pipeline order")
    cascade.add_argument("--roles", nargs="+", default=list(_CASCADE_ROLES),
                         choices=list(_CASCADE_ROLES),
                         help="Role per stage; the first must be 'translate'")
    cascade.add_argument("--glossary", required=True)
    cascade.add_argument("--glossary-stage", default="terminologise",
                         choices=list(_CASCADE_ROLES),
                         help="Which role is shown the dictionary (default: terminologise)")
    cascade.add_argument("--branches", default="")
    cascade.add_argument("--holdout", type=float, default=0.0)
    cascade.add_argument("--allow-mt-derived", action="store_true")
    cascade.add_argument("--max-terms", type=int, default=40)
    cascade.add_argument("--sample-size", type=int, default=0)
    cascade.add_argument("--seed", type=int, default=13)
    cascade.add_argument("--api-key")
    cascade.add_argument("--device")
    cascade.add_argument("--max-new-tokens", type=int, default=2048)
    cascade.add_argument("--output", required=True)
    cascade.add_argument("--summary")
    cascade.set_defaults(handler=command_cascade)

    judge = commands.add_parser(
        "judge",
        help="Audit translations with an LLM clinical rubric (open-class error spans)",
    )
    judge.add_argument("--input", required=True, help="JSONL with source and hypothesis fields")
    judge.add_argument("--judge", required=True, help="Backend, e.g. api:DeepSeek-V4-Flash")
    judge.add_argument("--src-key", default="src_text")
    judge.add_argument("--hyp-key", default="hyp_text")
    judge.add_argument("--src-lang", default="de")
    judge.add_argument("--tgt-lang", default="en")
    judge.add_argument("--system-under-test",
                       help="Refuse to run if the judge is this model (self-evaluation guard)")
    judge.add_argument("--sample-size", type=int, default=0)
    judge.add_argument("--api-key")
    judge.add_argument("--device")
    judge.add_argument("--max-new-tokens", type=int, default=1536)
    judge.add_argument("--output", required=True)
    judge.add_argument("--summary")
    judge.set_defaults(handler=command_judge)

    roundtrip = commands.add_parser(
        "roundtrip",
        help="Translate back and forth for N cycles, scoring every step",
    )
    _add_data_args(roundtrip)
    _add_model_args(roundtrip)
    roundtrip.add_argument("--cycles", type=int, default=10,
                           help="Cycles; each is 2 passes (default 10 = 20 passes)")
    roundtrip.add_argument("--sample-size", type=int, default=0,
                           help="Length-stratified subsample; 0 = whole corpus")
    roundtrip.add_argument("--seed", type=int, default=13)
    roundtrip.add_argument("--term-bank")
    roundtrip.add_argument("--output", required=True)
    roundtrip.add_argument("--summary")
    roundtrip.set_defaults(handler=command_roundtrip)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    return 2  # argparse.error exits; this keeps type checkers happy.


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
