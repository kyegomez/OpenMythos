"""
Command-line interface for the OpenMythos DJ engine.

    # Plan a mix from a library in a given style
    python -m open_mythos.dj plan ./my_tracks/ --profile melodic_journey --length 12
    python -m open_mythos.dj plan tracks.csv --profile peak_time_techno --mixlang

    # Analyze a library and dump metadata as CSV
    python -m open_mythos.dj analyze ./my_tracks/ --out library.csv

    # Learn a DJ's personality from their setlists, save the profile + corpus
    python -m open_mythos.dj learn set1.txt set2.txt --name my_dj \\
        --out my_dj.json --corpus my_dj_corpus.txt

Audio folders need the optional deps:  pip install "open-mythos[audio]"
"""

from __future__ import annotations

import argparse
import csv
import sys

from .analysis import analyze_library
from .planner import plan_mix
from .profile import BUILTIN_PROFILES, DJStyleProfile, get_profile


def _load_profile(name: str) -> DJStyleProfile:
    if name in BUILTIN_PROFILES:
        return get_profile(name)
    # Otherwise treat it as a path to a profile JSON.
    return DJStyleProfile.from_json(name)


def _cmd_plan(args) -> int:
    library = analyze_library(args.source)
    profile = _load_profile(args.profile)
    plan = plan_mix(library, profile, length=args.length)
    if args.mixlang:
        from .mixlang import plan_to_mixlang

        print(plan_to_mixlang(plan, dj_name=profile.name))
    else:
        print(plan.to_text())
    return 0


def _cmd_analyze(args) -> int:
    library = analyze_library(args.source)
    cols = ["title", "artist", "bpm", "key", "energy", "duration", "genre", "path"]
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for t in library:
                w.writerow({c: getattr(t, c) for c in cols})
        print(f"Wrote {len(library)} tracks to {args.out}")
    else:
        for t in library:
            print(f"{t.label():40}  {t.bpm:6.1f} BPM  {t.key_name:5}  e{t.energy:.2f}")
    return 0


def _cmd_render(args) -> int:
    from .render import render_mix, write_playlist

    library = analyze_library(args.source)
    profile = _load_profile(args.profile)
    plan = plan_mix(library, profile, length=args.length)

    if args.playlist:
        write_playlist(plan, args.playlist)
        print(f"Wrote playlist -> {args.playlist}")
    result = render_mix(
        plan,
        args.out,
        target_bpm=args.bpm,
        beatmatch=not args.no_beatmatch,
    )
    bpm_note = f" (target {result.target_bpm:.0f} BPM)" if result.target_bpm else ""
    print(f"Rendered {result.duration_seconds/60:.1f} min @ {result.sample_rate} Hz"
          f"{bpm_note} -> {result.out_path}\n")
    print(result.cue_sheet())
    return 0


def _cmd_identify(args) -> int:
    from .identify import build_identifier, scan_mix, setlist_to_dict

    identifier = build_identifier(args.provider)

    def _prog(offset, res):
        mm, ss = divmod(int(offset), 60)
        who = f"{res.get('artist','')} - {res.get('title','')}" if res else "(no match)"
        print(f"  {mm:02d}:{ss:02d}  {who}")

    setlist = scan_mix(
        args.audio, identifier, name=args.name,
        segment_seconds=args.segment, hop_seconds=args.hop,
        delay=args.delay, on_progress=_prog,
    )
    print(f"\nIdentified {len(setlist.tracks)} distinct track(s).")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(setlist_to_dict(setlist), fh, indent=2)
    print(f"Wrote setlist -> {args.out}\n"
          f"Next: python -m open_mythos.dj learn {args.out} --name {args.name} "
          f"--enrich musicbrainz")
    return 0


def _cmd_enrich(args) -> int:
    from .enrich import JsonCache, build_provider, enrich_tracks

    library = analyze_library(args.source)
    provider = build_provider(args.provider)
    cache = JsonCache(args.cache) if args.cache else None
    hits = 0

    def _prog(tr, res):
        nonlocal hits
        mark = "ok " if res else "-- "
        if res:
            hits += 1
        print(f"  {mark}{tr.label():40}  {tr.bpm:6.1f} BPM  {tr.key_name:5}  e{tr.energy:.2f}")

    enrich_tracks(library, provider, cache=cache, delay=args.delay, on_progress=_prog)
    print(f"\nEnriched {hits}/{len(library)} tracks.")

    cols = ["title", "artist", "bpm", "key", "energy", "duration", "genre", "path"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for t in library:
            w.writerow({c: getattr(t, c) for c in cols})
    print(f"Wrote enriched library -> {args.out}")
    return 0


def _cmd_learn(args) -> int:
    import os

    from .setlist import (
        learn_profile_from_setlists,
        setlist_from_folder,
        setlist_from_json,
        setlist_from_text,
        setlist_from_tracklist,
        setlists_to_corpus,
    )

    setlists = []
    for i, path in enumerate(args.setlists):
        nm = f"{args.name}_{i + 1}"
        if os.path.isdir(path):
            setlists.append(setlist_from_folder(path, name=nm))
        elif path.lower().endswith(".json"):
            setlists.append(setlist_from_json(path, name=nm))
        else:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            if args.tracklist:
                setlists.append(setlist_from_tracklist(
                    text, name=nm, title_first=not args.artist_first))
            else:
                setlists.append(setlist_from_text(text, name=nm))

    if args.enrich:
        from .enrich import JsonCache, build_provider, enrich_tracks

        provider = build_provider(args.enrich)
        cache = JsonCache(args.cache) if args.cache else None
        filled = 0
        for sl in setlists:
            def _prog(tr, res):
                nonlocal filled
                if res:
                    filled += 1
            enrich_tracks(sl.tracks, provider, cache=cache,
                          delay=args.delay, on_progress=_prog)
        print(f"Enriched metadata for {filled} track(s).", file=sys.stderr)

    profile = learn_profile_from_setlists(setlists, name=args.name)
    print(profile.to_json())

    if args.out:
        profile.to_json(args.out)
        print(f"\nSaved profile -> {args.out}", file=sys.stderr)
    if args.corpus:
        rows = setlists_to_corpus(setlists, out_path=args.corpus)
        print(f"Wrote {len(rows)} training row(s) -> {args.corpus}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="open_mythos.dj", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("plan", help="plan a mix from a library")
    pp.add_argument("source", help="folder of audio, CSV, or JSON list of tracks")
    pp.add_argument("--profile", default="melodic_journey",
                    help="built-in profile name or path to a profile JSON")
    pp.add_argument("--length", type=int, default=None, help="number of tracks")
    pp.add_argument("--mixlang", action="store_true",
                    help="emit the mix-language token string instead of text")
    pp.set_defaults(func=_cmd_plan)

    pa = sub.add_parser("analyze", help="analyze a library's metadata")
    pa.add_argument("source", help="folder of audio, CSV, or JSON list of tracks")
    pa.add_argument("--out", help="write metadata to this CSV path")
    pa.set_defaults(func=_cmd_analyze)

    pr = sub.add_parser("render", help="render a mix to an audio file")
    pr.add_argument("source", help="folder of audio, CSV, or JSON list of tracks")
    pr.add_argument("--out", required=True, help="output audio path (.wav)")
    pr.add_argument("--profile", default="melodic_journey",
                    help="built-in profile name or path to a profile JSON")
    pr.add_argument("--length", type=int, default=None, help="number of tracks")
    pr.add_argument("--bpm", type=float, default=None,
                    help="beatmatch target BPM (default: median of tracks)")
    pr.add_argument("--no-beatmatch", action="store_true",
                    help="do not time-stretch tracks to a common tempo")
    pr.add_argument("--playlist", help="also write an M3U playlist here")
    pr.set_defaults(func=_cmd_render)

    pi = sub.add_parser("identify",
                        help="recognize tracks in a mix's audio -> setlist JSON")
    pi.add_argument("audio", help="path to the mix audio file (you supply it)")
    pi.add_argument("--provider", required=True,
                    help="'shazam' (no key) | 'audd:API_TOKEN'")
    pi.add_argument("--out", required=True, help="write the setlist JSON here")
    pi.add_argument("--name", default="scanned_mix", help="name for the setlist")
    pi.add_argument("--segment", type=float, default=20.0,
                    help="recognition clip length in seconds (default 20)")
    pi.add_argument("--hop", type=float, default=60.0,
                    help="seconds between clips sampled (default 60)")
    pi.add_argument("--delay", type=float, default=1.0,
                    help="seconds between API calls (default 1.0)")
    pi.set_defaults(func=_cmd_identify)

    pe = sub.add_parser("enrich", help="fill missing bpm/key/energy metadata")
    pe.add_argument("source", help="folder of audio, CSV, or JSON list of tracks")
    pe.add_argument("--provider", required=True,
                    help="'musicbrainz' | 'getsongbpm:KEY' | 'csv:PATH' | 'a+b'")
    pe.add_argument("--out", required=True, help="write the enriched library CSV here")
    pe.add_argument("--cache", help="JSON cache path for lookups")
    pe.add_argument("--delay", type=float, default=1.0,
                    help="seconds between network lookups (default 1.0)")
    pe.set_defaults(func=_cmd_enrich)

    pl = sub.add_parser("learn", help="learn a DJ profile from setlists")
    pl.add_argument("setlists", nargs="+",
                    help="setlist files (.txt/.json) or folders of ordered audio")
    pl.add_argument("--name", required=True, help="name for the learned profile")
    pl.add_argument("--out", help="write the learned profile JSON here")
    pl.add_argument("--corpus", help="write an RDT training corpus here")
    pl.add_argument("--tracklist", action="store_true",
                    help="parse inputs as pasted tracklists (indexes/timestamps)")
    pl.add_argument("--artist-first", action="store_true",
                    help="with --tracklist: 'A - B' means Artist - Title")
    pl.add_argument("--enrich", metavar="SPEC",
                    help="fill missing bpm/key/energy, e.g. 'musicbrainz', "
                         "'getsongbpm:KEY', 'csv:known.csv', 'csv:k.csv+musicbrainz'")
    pl.add_argument("--cache", help="JSON cache path for enrichment lookups")
    pl.add_argument("--delay", type=float, default=1.0,
                    help="seconds between network lookups (default 1.0)")
    pl.set_defaults(func=_cmd_learn)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
