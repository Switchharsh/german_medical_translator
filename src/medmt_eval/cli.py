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
    output = translate_segments(translator, segments)
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
    predictions = translate_segments(translator, segments)
    if args.predictions_output:
        write_jsonl(predictions, args.predictions_output)
    evaluations, summary = evaluate_hypotheses(
        segments,
        [str(row["hyp_text"]) for row in predictions],
        model=translator.name,
        generation=translator.generation_config,
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


def _add_data_args(parser: argparse.ArgumentParser, *, allow_reverse: bool = True) -> None:
    parser.add_argument("--input", required=True, help="JSONL, CSV, TSV, or Parquet input")
    parser.add_argument("--src-lang", help="Default source language when input omits it (en/de)")
    parser.add_argument("--tgt-lang", help="Default target language when input omits it (en/de)")
    if allow_reverse:
        parser.add_argument("--reverse", action="store_true", help="Reverse source/reference for DE↔EN evaluation")


_MODEL_CHOICES = ["identity", "opus", "nllb", "madlad", "tower", "deepl", "prompted-llm", "hymt2", "translategemma"]


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=_MODEL_CHOICES)
    parser.add_argument("--model-id", help="Override the adapter's default Hugging Face model ID")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", help="Torch device, for example cuda or cpu")
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
