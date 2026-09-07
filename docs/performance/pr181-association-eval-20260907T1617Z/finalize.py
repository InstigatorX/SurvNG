"""Package sanitized offline results, preserving raw inputs and original blocked evidence."""
import json,csv,hashlib,subprocess,zipfile,os,re,statistics
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
ORIGINAL=ROOT.parent
SHA='67e67ad73b7dc950dd55f15cabd4cb6d8f35531d'
def read(path):return json.loads(path.read_text())
def save(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def command(args):return subprocess.check_output(args,text=True,timeout=20).strip()
def safe(value):
    if isinstance(value,dict):
        return {k:safe(v) for k,v in value.items() if k not in ('_tracking_embedding','embedding','embeddings','tracking_config','config','api_auth','web_auth','token','token_hash','authorization','segments','frames') or (k=='frames' and not isinstance(v,list))}
    if isinstance(value,list):return [safe(v) for v in value]
    if isinstance(value,str):
        return value.replace(str(ORIGINAL),'SCRATCH').replace('SERVICE_VENV','SERVICE_VENV').replace('DEPLOYMENT','DEPLOYMENT')
    return value
def main():
    os.umask(0o077)
    summary=read(ROOT/'summary.json');timing=read(ROOT/'timing-summary.json');corpus=read(ROOT/'corpus.json');validation=read(ROOT/'private/replay-validation.json')
    initial=read(ROOT/'private/runtime-resume-before.json');last=json.loads(command(['DEPLOYMENT/survngctl','status','--compact','--socket','/run/survng/observability.sock']))
    save(ROOT/'private/runtime-final.json',last)
    cfgpath=Path('DEPLOYMENT/config.json');cfg=read(cfgpath);oldsettings=read(ORIGINAL/'private/saved-settings-selected.json')['tracking_saved_not_effective']
    cfgsha=digest(cfgpath);oldmeta=read(ORIGINAL/'private/metadata-before.json')
    preservation={'deployed_checkout_sha':command(['git','-C','DEPLOYMENT','rev-parse','HEAD']),'deployed_git_status':command(['git','-C','DEPLOYMENT','status','--short']),
                  'pid_before':initial['process']['pid'],'pid_after':last['process']['pid'],'instance_before':initial['process']['instance_id'],'instance_after':last['process']['instance_id'],
                  'original_blocked_run_config_sha256':oldmeta['config_sha256'],'resumed_observed_config_sha256':'49b26dd91971e63175027da8c78a7b85da16dbe0f777fb8992dcaa036314a8ac','final_config_sha256':cfgsha,
                  'config_file_mtime_utc':datetime.fromtimestamp(cfgpath.stat().st_mtime,timezone.utc).isoformat(),'saved_tracking_subtree_unchanged_since_original':oldsettings==cfg.get('detector',{}).get('tracking',{}),
                  'external_configuration_change':'Fingerprint changed between original blocked run and resumed execution; mtime 16:26:09Z precedes resume/capture. User supplied an API token during this interval. Full original configuration was deliberately not saved, so not all changed fields can be reconstructed or attributed to token creation.',
                  'task_writes':'Only scratch files and authorized normal Compare cache/history writes. No deployment/configuration edits, restart, or settings changes performed by this task.',
                  'original_archive_sha256_now':digest(ORIGINAL/'shareable-results.zip')}
    before=read(ROOT/'private/history-resume-before.json');after=read(ROOT/'private/history-resume-after.json')
    preservation['all_preexisting_comparison_rows_preserved_by_full_content_hash']=all(r in after for r in before)
    preservation['new_comparison_rows']=[r for r in after if r['id'] not in {x['id'] for x in before}]
    save(ROOT/'preservation.json',preservation)
    snapshots=[initial,*[read(p) for p in sorted((ROOT/'private').glob('capture-*.health-*.json'))],read(ROOT/'private/runtime-resume-after.json')]
    def signature(e):return (e['timestamp'],e['level'],e['logger'],e['message'])
    original_logs={signature(e) for e in initial['recent_logs']['entries']}
    new_logs={signature(e) for s in snapshots for e in s['recent_logs']['entries'] if signature(e) not in original_logs}
    resources={'snapshot_count':len(snapshots),'min_available_memory_gib':min(s['experiment_os']['memory_available']/2**30 for s in snapshots),
               'memory_floor_gib':1.6,'connected_counts':sorted({sum(c['connected'] for c in s['cameras']) for s in snapshots}),
               'enabled_recording_active_counts':sorted({sum(c['recording_enabled'] and c['recording'] for c in s['cameras']) for s in snapshots}),
               'queue_depths':sorted({s['detector']['runtime']['queue_depth'] for s in snapshots}),'failed_inferences':sorted({s['detector']['runtime']['failed_inferences'] for s in snapshots}),
               'detector_counter_before':initial['detector']['runtime']['total_inferences'],'detector_counter_after':snapshots[-1]['detector']['runtime']['total_inferences'],
               'cpu_psi_avg10_max_percent':max(float(re.search(r'avg10=([0-9.]+)',s['experiment_os']['cpu_pressure'])[1]) for s in snapshots),
               'memory_psi_avg10_max_percent':max(float(re.search(r'avg10=([0-9.]+)',s['experiment_os']['memory_pressure'])[1]) for s in snapshots),
               'new_log_entries':[{'timestamp':t,'level':l,'logger':g,'message':m} for t,l,g,m in sorted(new_logs)],
               'coverage_limit':'Before/after capture snapshots, not continuous monitoring. CPU PSI measures CPU waiting pressure, NOT CPU utilization. Service average CPU was not sampled; no fleet CPU/iGPU savings claim.'}
    save(ROOT/'resource-summary.json',resources)
    settings=validation[0]
    replay=read(ROOT/'private/capture-63156.replay.json');keys=('implementation','sample_fps','adaptive_sampling_enabled','stable_sample_fps','min_confirmations','lost_timeout_seconds','reid_enabled','reid_match_threshold','reid_max_age_seconds','reid_max_embeddings_per_frame','vehicle_reid_enabled','vehicle_reid_match_threshold','deferred_reid_enabled')
    effective={k:replay['tracking_config'].get(k) for k in keys}
    sources={str(p.relative_to(ORIGINAL/'checkout')):digest(p) for p in [ORIGINAL/'checkout/survng/app/tracking_comparison.py',ORIGINAL/'checkout/survng/app/tracking_evaluation.py',ORIGINAL/'checkout/survng/app/object_track/hybrid.py',ORIGINAL/'checkout/survng/app/object_track/multicue.py']}
    cache_files=[]
    for case in corpus['selected']:
        for media in ('/mnt/media1/SurvNG','/mnt/media2/SurvNG'):
            directory=Path(media)/'event_clips'/case['camera_id']/'main'
            if directory.is_dir():
                for path in directory.glob(f'{case["event_id"]}-0-30000-a3-*.mp4'):
                    cache_files.append({'event_id':case['event_id'],'name':path.name,'bytes':path.stat().st_size,'mtime_utc':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()})
    result_hashes={p.name:digest(p) for p in sorted((ROOT/'results').glob('*.json'))}
    manifest={'verdict':'NO_OBSERVED_EFFECT','verdict_scope':'No observed effect on confirmed association outputs, not on cue scores or all timing.',
              'confidence':{'execution':'high','confirmed_output_agreement':'high','identity_accuracy':'unknown','timing_selected_case':'moderate','scenario_coverage':'low to moderate'},
              'finished_utc':datetime.now(timezone.utc).isoformat(),'initial_task_start_utc':'2026-09-07T16:16:00Z','resumed_runtime_snapshot_utc':initial['generated_at'],'corpus_locked_utc':corpus['locked_utc'],
              'candidate_sha':SHA,'candidate_branch':'experiment/hybrid-tracktrack-association-cues','remote_head_verified':True,'remote_pr_state':'OPEN',
              'experimental_baseline':'HybridObjectTracker / survng_hybrid from same isolated experiment checkout, including corrected PR #180 baseline; parent b131688 merges #180.',
              'candidate':'HybridMultiCueObjectTracker / survng_hybrid_multicue from isolated experiment checkout; no TAI/new-track suppression or weight tuning.',
              'deployed_revision_evidence':'Clean checkout SHA matches candidate head; service cwd DEPLOYMENT and start after its update. No in-process build fingerprint exists, so checkout SHA is not independent proof of loaded bytes.',
              'runtime':{'service':'survng.service','pid':last['process']['pid'],'instance_id':last['process']['instance_id'],'service_start_utc':'2026-09-07T15:26:57Z','final_uptime_seconds':last['process']['uptime_seconds'],'cwd':'DEPLOYMENT','interpreter':'SERVICE_VENV/bin/python','base_url':'http://127.0.0.1:8088','base_path':'/survng','timezone':'America/New_York (EDT UTC-04:00)','socket':'/run/survng/observability.sock'},
              'versions':oldmeta['versions'],'python':oldmeta['python'],'opencv_threads':oldmeta['cv2_threads'],'runtime_observable_tracking_settings':initial['tracking']['settings'],
              'capture_in_memory_tracking_setting_allowlist':effective,'settings_provenance':'Compare serializes the running tracking config into each checksummed replay; all six tracking configs compare equal. Full configs are private/excluded from archive.',
              'selections':safe(corpus['selected']),'missing':corpus['missing'],'exclusions':corpus['exclusions'],'reserves':[],'discovery':corpus['discovery'],
              'captures':safe(validation),'sources_sha256':sources,'result_sha256':result_hashes,'preservation':preservation,'resources':resources,
              'permitted_side_effects':{'new_comparison_ids':[r['id'] for r in preservation['new_comparison_rows']],'old_rows_preserved':preservation['all_preexisting_comparison_rows_preserved_by_full_content_hash'],'event_clip_cache_files':cache_files,
                  'concat_temp_files':'Normal service clip generation creates/removes concat manifests and temporary MP4s per reviewed source; not independently inventoried.','no_verdict_changes':True,'no_other_application_mutations_requested':True},
              'regressions':{'original_gate':'2 import collection errors due installed tests namespace; original log preserved','approved_scratch_adjustment':'Added only checkout/tests/__init__.py package marker; no assertions or tracker code changed','rerun':'64 passed,64 subtests passed,exit0,.47s'},
              'review_models':'gpt-6-astra agent astra_association_review directly reviewed candidate/evaluation source, analyzer, 13 result/agreement files and timing artifacts; parent executed bounded measurement work.',
              'identity_metrics':{'IDF1':'NOT MEASURED','true_identity_switches':'NOT MEASURED','true_false_merges':'NOT MEASURED'},
              'raw_replays':'Unmodified private/capture-EVENT.replay.json, full response private/capture-EVENT.json. Raw embeddings and full configs deliberately excluded from shareable archive.',
              'token_handling':'Operator token held only in API-session memory; no credential file or result contained it. Session closed after six captures.'}
    assert all(read(ROOT/'private'/f'capture-{c["event_id"]}.replay.json')['tracking_config']==replay['tracking_config'] for c in corpus['selected'])
    assert all(digest(ROOT/'private'/f'capture-{v["event_id"]}.replay.json')==v['file_sha256'] for v in validation)
    assert preservation['all_preexisting_comparison_rows_preserved_by_full_content_hash']
    assert preservation['original_archive_sha256_now']=='d46f069cfec5f5b51cc37fd8e5ed2577dc48a53e750a7e54045dff98858bfff8'
    save(ROOT/'manifest.json',safe(manifest))
    summary.update(verdict=manifest['verdict'],verdict_scope=manifest['verdict_scope'],confidence=manifest['confidence'],timing_repeat=timing)
    save(ROOT/'summary.json',summary)
    report='''# PR #181 saved-detection association evaluation

## Executive verdict

**NO_OBSERVED_EFFECT on confirmed association outputs.** All 13 executed pairs produced literally identical confirmed frame observations and final track summaries. Cue scores changed in 7 profiles across 3 distinct incidents, but no changed confirmed assignment or continuity decision was observed. This does not establish identity accuracy or justify production promotion.

Execution/output-agreement confidence: **high**. Scene-category confidence: **low to moderate**. Selected-case timing confidence: **moderate**. Identity accuracy: **unknown**.

Completed **13/14 requested paired evaluations**, across six distinct incidents and four cameras: all six at fixed_2fps and fixed_075fps, plus one sparse_gaps case. The second suitable gap case was not established; it was not replaced by an inactive clip. There were six sequential capture attempts, all successful, no retries or reserve captures. The approved scratch-only test-package marker resolved the import collision: **64 tests and 64 subtests passed**. The original failed log and blocked-run archive remain unchanged.

## Experiment and runtime provenance

Candidate and experimental baseline are both from isolated source at `67e67ad73b7dc950dd55f15cabd4cb6d8f35531d` (PR #181 current open head, verified twice). Baseline is corrected `survng_hybrid` / `HybridObjectTracker`; candidate is `survng_hybrid_multicue` / `HybridMultiCueObjectTracker`. Parent `b131688` merges PR #180. No tracking math, weights, assertions, production registry or new-track policy was changed. No TAI was implemented.

Live `survng.service` PID **3257018**, owner instance `4aafae2f5f4ffd06d624088b`, cwd `DEPLOYMENT`, uses `SERVICE_VENV/bin/python` and Uvicorn. Start: **2026-09-07 15:26:57 UTC**, after the checkout update. The clean deployed checkout has the same SHA; an in-process build hash is unavailable, so this is deployment evidence, not independent loaded-byte proof. Local API is `/survng` on `127.0.0.1:8088`; owner-only observability socket `/run/survng/observability.sock` worked. Deployment/camera time is America/New_York (UTC-04:00); measurements use UTC. Camera timezone overrides were unset; viewed timestamps corroborate the deployment offset.

Python 3.12.3, NumPy 2.4.6, OpenCV 5.0.0.93, Pydantic 2.13.4, pytest 9.1.1; dependency metadata is in the manifest. Existing OpenCV default: 16 threads, unchanged. Imports resolved into scratch, not the deployed checkout. No packages or optional runtimes were installed. The service's optional upstream trackers were unavailable, but that did not invalidate the full replay bundles or either Hybrid engine.

The running Compare capture serialized identical effective tracking configurations into all six replays. Relevant settings: nominal 3fps, adaptive stable 0.75fps, lost timeout 4s, min confirmations 2, person and vehicle ReID enabled; detailed allowlisted values are in the manifest. Both engines received identical selected detections, timestamps, dimensions, settings, initialization and supplied embeddings. The replay does not measure live selective-ReID scheduling or total accelerator savings.

## Locked corpus and selection limits

Selection was locked at '''+corpus['locked_utc']+''', before capture or paired outcome inspection. Metadata selected candidates; modest local image inspection checked scene plausibility. It did not create identity labels. “Crossing” slots are suspected close interactions, not proven geometric crossings. The occlusion slot has partial car/door obstruction; a true same-identity return is unproven. Historical tracking sessions were **terminal/interrupted**, commonly stale handoff, not successfully completed tracking coverage. The selected incidents are days old, with no active tracking-job rows and finalized retained media.

| Slot | Camera / actual child event | UTC anchor | Rationale / confidence |
|---|---|---|---|
'''
    for c in corpus['selected']:report+=f"| {c['slot']} | {c['camera_id']} / {c['event_id']} | {c['timestamp']} | {c['rationale']} ({c['selection_confidence']}) |\n"
    report+='''
The original 46 shortlisted details were reused; 11 additional detail requests included one 422 from omitting intermediate child IDs, corrected using the actual full child list. Two 100-item API summary pages were read; the missing categories were widened from 72 hours to seven days, not 30 days. No arbitrary unlimited search occurred. Gate event 65540 was excluded after its “two people” proved to be truck occupants. Child 64898 already had comparison 70 and was not overwritten; earlier child 64897 anchors the same incident's interaction inside the following 30 seconds. Earlier candidate 66002 had an indexed one-second gap and was not used. No post-result substitutions occurred.

All six requested 30-second main-stream windows have indexed continuous coverage and retained files. Each normal Compare capture decoded **90 source-PTS frames** (nominal raw sampling 3fps, not 2fps); all timestamps are strictly increasing and checksums valid. There were 540 captured frames, 632 detections, and 438 supplied embeddings. Actual raw intervals span approximately 0.15–0.49s; per-case cadence and digests are in the manifest. Selected profiles have 60 frames at about 1.99fps, 23 at about 0.74–0.75fps, and 40 at 1.315fps for the one gap case. Sampling only selects existing observations; it cannot manufacture frames. The source-PTS label refers to the exported clip; sparse original-segment image seeks used for scene selection are approximate and were not treated as frame-exact identity labels.

Gap case 64897 contains person observations before, inside and after both imposed gaps, and beyond second 21. At elapsed [21,24) there are 9 raw frames and 18 person detections; [27,30) contains 8 frames and 8 person detections. Profiles drop [5,9) and [15,21), rather than inserting empty detections. Gaps exceeding retention can legitimately create different IDs. This offline harness continues beyond the normal lost-track predicate and is not the complete production lifecycle.

## Per-case/profile results

Each paired value below is **baseline / candidate**. Tracks are confirmed-summary count; births are all new-track diagnostic allocations, including tentative tracks. Observations count confirmed outputs. Fragmentation is the existing count-minus-maximum-simultaneous proxy, not true fragmentation. ReID counts supplied-embedding recoveries, not independent identities. All engines executed successfully. The CSV contains all fields, errors, exact timing deltas/ratios and agreement counts.

| Event / profile | Frames | Tracks | Births | Observations | Fragment proxy | ReID recoveries | Embeddings | Tracker ms/frame B/C | Delta ms / ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
    for r in summary['rows']:
        pair=lambda k:f"{r['baseline_'+k]}/{r['candidate_'+k]}"
        report+=f"| {r['event_id']} / {r['profile']} | {r['frames']} | {pair('confirmed_track_count')} | {pair('new_track_count')} | {pair('confirmed_observations')} | {pair('fragmentation_proxy')} | {pair('reid_recoveries')} | {pair('supplied_embeddings')} | {pair('average_ms_per_frame')} | {r['absolute_delta_ms']:+.3f} / {r['candidate_baseline_ratio']:.2f}x |\n"
    report+='''
| Event / profile | Valid cue pairs | Ambiguous pairs | Adjusted pairs | Direction available | Confidence available | Changed confirmed frames |
|---|---:|---:|---:|---:|---:|---:|
'''
    for r in summary['rows']:report+=f"| {r['event_id']} / {r['profile']} | {r['valid_cue_pairs']} | {r['ambiguous_cue_pairs']} | {r['adjusted_cue_pairs']} | {r['direction_available']} | {r['confidence_available']} | {r['changed_frames']} |\n"
    report+='''
These counters cover evaluated pairs across the replay, not necessarily selected assignments. Unambiguous strict-geometry matches keep their baseline scores. HMIoU, confidence and direction penalties were exercised where available; no cue adjustment changed a confirmed decision here. Direction/confidence availability uses elapsed timestamps, becomes neutral for stale gaps and can be absent with insufficient motion/history. The one sparse case's five tracks / fragmentation proxy two and three ReID recoveries are identical in both engines; they are not proven errors.

The two vehicle cases and single-person control have **zero ambiguous/adjusted cue pairs** in both regular profiles. These are untouched controls for association-score changes, not evidence about ambiguous-cue quality. Tentative tracks are absent from frame_observations. Whole-track alignment used shared exact input observations and one-to-one maximum-weight matching, preserving split/merge disagreements. Same-class boxes with IoU >=0.95 were marked ambiguous rather than forced; none occurred in these outputs. All outputs and track summaries were also literally equal before ID normalization. No changed timestamps or material disagreement episodes exist, so no observer reruns or diagnostic overlays were manufactured. The analyzer's row-max tie flag is not a general uniqueness test for global assignments; literal equality makes that limitation immaterial here.

**IDF1: NOT MEASURED. True identity switches: NOT MEASURED. True false merges: NOT MEASURED.** There are no independent exact-replay labels. Fewer tracks, equal baseline IDs, stored histories and embeddings cannot establish accuracy. No detection-recall conclusion is drawn from these unlabelled recordings.

## Timing: small selected-case overhead, not fleet CPU

Initial timing is one baseline-then-candidate observation per profile, including detection deepcopy and tracker update wall time, excluding initialization. Ratios are unstable at these small absolute costs. The largest positive initial delta was event 63156/fixed_2fps: +0.060 ms/frame. It was checked with one excluded warmup and three bounded measured repetitions, alternating engine order C/B, B/C, C/B:

| Round | Baseline ms/frame | Candidate ms/frame | Paired delta ms |
|---|---:|---:|---:|
'''
    for t in timing['runs'][1:]:report+=f"| {t['round']} | {t['baseline_ms_per_frame']:.3f} | {t['candidate_ms_per_frame']:.3f} | {t['candidate_ms_per_frame']-t['baseline_ms_per_frame']:+.3f} |\n"
    report+='''
Baseline median **0.167**, candidate median **0.185 ms/frame**: difference of medians **+0.018 ms**, about +10.8%. Median paired delta is **+0.017 ms**; paired range +0.009 to +0.020 ms. Baseline range 0.167–0.168, candidate 0.176–0.187. All repeated observations remained equal. This supports a small observed cost in this selected case, not a statistically established/general regression or a fleet CPU/iGPU change. There is no measured association benefit to offset it in this corpus. Apparent speedups in other one-shot rows are not claimed as improvements. No long soak or inference rerun was performed.

## Live safety, health and side effects

'''
    report+=f"{resources['snapshot_count']} before/after snapshots show 13 connected cameras and 11 enabled recordings active throughout sampled checks; detector queue depth and failed-inference count remained zero. The sampled minimum available memory was approximately **{resources['min_available_memory_gib']:.2f} GiB**, above the 1.6 GiB floor. CPU PSI avg10 peaked at {resources['cpu_psi_avg10_max_percent']:.2f}%; memory PSI avg10 stayed {resources['memory_psi_avg10_max_percent']:.2f}%. PSI is waiting pressure, not CPU utilization. No continuous service-CPU percentage series was collected, and snapshot health cannot rule out every transient issue.\n\n"
    report+='''No new recorder/detector failures appeared in the available recent-log snapshots. Old startup stream-open failures around 15:27 UTC and historical interrupted tracking predate this experiment. New log entries show the six normal QSV event-clip builds, not service failures. All six captures completed with source-PTS frames and zero appearance failures. Optional upstream tracker errors did not trigger runtime installation or input fabrication.

Only authorized capture effects were observed: new comparisons **77–82**, six normal 30-second event-clip cache files, and the associated normal temporary clip work. Full pre-existing comparison content/verdict hashes were preserved; no row was overwritten or evicted, including row 70. The normal running database continued changing; it was not treated as static. Cache file sizes/timestamps and comparison provenance are in the manifest. Experiment processes ran nice 10 / idle IO where supported; that does not throttle service-side capture or guarantee iGPU isolation. No streams were added, auth bypassed, debug enabled, priorities changed on the service, packages installed, or independent inference worker started.

The service PID, instance and clean checkout SHA remained unchanged. Configuration preservation needs an explicit caveat: its fingerprint changed from the original blocked run's `5cad7961…` to `49b26dd9…`, with file mtime **16:26:09 UTC**, before the resumed snapshot and first capture. This coincides with the operator token being supplied. No task code wrote configuration; the saved tracking subtree is unchanged. The full original config was intentionally not retained, so not every external field difference can be reconstructed. The resumed observed fingerprint matches the final one. This is an external change, not an assertion that all live configuration stayed byte-identical across the user's authorization interval.

## Recommendation and artifacts

Keep the candidate offline; do not promote or tune it on this evidence. The smallest justified next evaluation is a corpus selected independently for association ambiguity and identity-labelled before comparing tracker outcomes, without selecting cases based on candidate wins or disagreements. No runtime fix, architectural expansion, or TAI work is justified by this test. The current sample demonstrates successful execution and confirmed-output agreement on these replays, not superior association quality.

Raw inputs and full Compare responses remain private locally. The ZIP contains sanitized report, summaries, manifest, per-pair and repeated-timing results, agreement records, bounded logs and scripts—no raw embeddings, full configs, DBs, model weights or full recordings. `review/` records why there are no disagreement overlays. Reproduction commands are in `commands.md`; raw replays must be supplied privately to reproduce. Astra (`gpt-6-astra`, agent `astra_association_review`) directly reviewed the relevant source, analyzer, all 13 result/agreement files and repeat-timing evidence. No GitHub write or PR was made. The API-session process was closed, and its in-memory token was not persisted; revoke the token shared in chat when finished.
'''
    (ROOT/'report.md').write_text(report)
    (ROOT/'review/README.md').write_text('# Review artifacts\n\nNo material confirmed-output disagreement occurred in 13 pairs. No synthetic disagreement episode or overlay was generated. `episodes.json` is empty. Modest selection contact sheets were visually inspected in private scratch; they are not identity labels and are excluded from sharing. Raw frame observations and track summaries are literally equal in each pair.\n')
    # Exports only explicit allowlisted artifacts, never private files or source config.
    paths=[ROOT/n for n in ('report.md','summary.csv','summary.json','manifest.json','preservation.json','resource-summary.json','executions.json','timing-summary.json','commands.md','api_session.py','run_pairs.py','analyze_pairs.py','timing_repeat.py','media_check.py','selection_extra.py','lock_corpus.py','finalize.py','offline_reproduce.py')]
    paths += sorted((ROOT/'results').glob('*.json'))+sorted((ROOT/'logs').glob('*.log'))+sorted((ROOT/'review').glob('*'))
    archive=ROOT/'shareable-results.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            if p.suffix=='.json':content=json.dumps(safe(read(p)),indent=2)+'\n'
            else:content=safe(p.read_text())
            assert not re.search(r'survng_[A-Za-z0-9_-]{35,}',content),'possible credential in sharing output'
            z.writestr(str(p.relative_to(ROOT)),content)
        for name in ('regression.log','regression-rerun.log'):
            z.writestr('logs/'+name,safe((ORIGINAL/'logs'/name).read_text()))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert not any(name.startswith('private/') for name in z.namelist())
    print(json.dumps({'report':str(ROOT/'report.md'),'archive':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':digest(archive),'preservation':preservation,'resources':resources},indent=2))
if __name__=='__main__':main()
